import json
import logging
import traceback
from pathlib import Path
import celery
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
import requests
import hashlib
# Components
from blueprint_brain.worker.celery_app import celery_app
from blueprint_brain.services.storage import StorageService
from blueprint_brain.config.settings import settings
# Engines
from blueprint_brain.src.utils.pdf_converter import PDFConverter
from blueprint_brain.src.inference.engine import InferenceEngine
from blueprint_brain.src.ocr.engine import OCREngine
from blueprint_brain.src.fusion.assembler import FusionAssembler
from blueprint_brain.src.utils.visualizer import Visualizer
from blueprint_brain.src.db.session import SessionLocal
from blueprint_brain.src.db import crud
import time
from contextlib import contextmanager
from blueprint_brain.src.monitoring.metrics import INFERENCE_DURATION, ROOMS_DETECTED

@contextmanager
def timer_logger(label: str, job_id: str):
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        # Log in JSON format for Datadog/ELK parsing
        logger.info(json.dumps({
            "metric": "task_duration",
            "job_id": job_id,
            "stage": label,
            "seconds": round(duration, 3)
        }))

logger = logging.getLogger(__name__)
storage = StorageService()

class ModelTask(Task):
    """
    Abstract Task class to handle ML Model resource management.
    """
    _vision_engine = None
    _ocr_engine = None

    @property
    def vision_engine(self):
        if self._vision_engine is None:
            self._vision_engine = InferenceEngine()
        return self._vision_engine

    @property
    def ocr_engine(self):
        if self._ocr_engine is None:
            self._ocr_engine = OCREngine()
        return self._ocr_engine

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Log full stack trace on failure for debugging"""
        logger.error(f"Task {task_id} failed: {exc}")
        logger.error(traceback.format_exc())

@celery_app.task(
    bind=True, 
    base=ModelTask, 
    name="blueprint_brain.worker.tasks.process_blueprint",
    soft_time_limit=600, # 10 minutes max per file
    acks_late=True # Only ack after success (prevents data loss)
)
def process_blueprint(self, file_key: str, webhook_url: str = None):
    job_id = self.request.id
    logger.info(f"[{job_id}] Processing Started: {file_key}")
    
    local_dir = Path(f"/tmp/{job_id}")
    local_dir.mkdir(parents=True, exist_ok=True)
    # Create DB Session
    db = SessionLocal()
    
    try:
        crud.update_job_status(db, job_id, "PROCESSING")
        # 1. Download
        local_input = local_dir / "input_file"
        storage.download_file(file_key, str(local_input))

        # --- OPTIMIZATION: Deduplication ---
        # Calculate Hash
        with open(local_input, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        # Check Cache (Redis)
        # We store: "hash_xyz" -> "results/old_job_id/data.json"
        import redis
        r = redis.from_url(settings.REDIS_URL)
        cached_result_key = r.get(f"file_hash:{file_hash}")
        
        if cached_result_key:
            cached_result_key = cached_result_key.decode('utf-8')
            logger.info(f"[{job_id}] Cache Hit! Found result from {cached_result_key}")
            # We can return immediately. 
            # Note: We might want to copy the artifacts to the new job_id folder in S3 
            # if we want strictly separated user data, but for efficiency, we reference existing.
            return {"status": "success", "result_key": cached_result_key, "cached": True}

        # 2. Conversion
        self.update_state(state='PROCESSING', meta={'progress': 10, 'status': 'Converting PDF...'})
        with timer_logger("pdf_conversion", job_id):
         images = PDFConverter.to_images(local_input) if file_key.endswith('.pdf') else [cv2.imread(str(local_input))]
        start_time = time.time()
        with timer_logger("gpu_inference_total", job_id):
          # 3. Analysis Loop
          final_pages = []
          fusion_engine = FusionAssembler()
          visualizer = Visualizer(settings.CLASS_MAP)

          for i, img in enumerate(images):
              # Calculate granular progress
              base_prog = 20
              per_page = 70 / len(images) # 70% of progress bar is analysis
              current_prog = int(base_prog + (i * per_page))
              
              self.update_state(state='PROCESSING', meta={
                  'progress': current_prog, 
                  'status': f'Analyzing Page {i+1}/{len(images)}'
              })

              # A. Inference (Using Cached Engines from self)
              vision_res = self.vision_engine.process_full_image(img)
              ocr_res = self.ocr_engine.analyze_image(img)

              # B. Prepare Detections
              detections = []
              if len(vision_res['boxes']) > 0:
                  for idx, box in enumerate(vision_res['boxes']):
                      cls_id = int(vision_res['classes'][idx])
                      detections.append({
                          "label": visualizer.id_to_name.get(cls_id, "Unknown"),
                          "bbox": box.tolist(),
                          "confidence": float(vision_res['scores'][idx])
                      })

              # C. Logic Fusion
              # (Generate pseudo-mask if needed, or use segmentation output)
              h, w = img.shape[:2]
              mock_mask = np.zeros((h, w), dtype=np.uint8) # Replace with real mask from vision_res if available
              
              page_data = fusion_engine.assemble_floorplan(img.shape, mock_mask, detections, ocr_res)
              
              # D. Artifact Generation
              annotated = visualizer.draw_bboxes(img.copy(), vision_res['boxes'], vision_res['scores'], vision_res['classes'])
              img_name = f"{job_id}_p{i+1}.jpg"
              img_path = local_dir / img_name
              cv2.imwrite(str(img_path), annotated)
              
              # Upload with Content-Type for browser viewing
              storage.upload_file(open(img_path, 'rb'), f"results/{job_id}/{img_name}", content_type="image/jpeg")
              
              page_data['page'] = i + 1
              page_data['image_key'] = f"results/{job_id}/{img_name}"
              final_pages.append(page_data)
        duration = time.time() - start_time
        INFERENCE_DURATION.labels(model_type="full_pipeline").observe(duration)
                # METRIC 2: Count Rooms
        total_rooms = 0
        for page in final_pages: # Assuming final_pages populated in loop
            # page['data'] is list of rooms
            room_count = len(page.get('data', []))
            total_rooms += room_count
            
        ROOMS_DETECTED.inc(total_rooms)
        # 4. Final Save
        output = {"job_id": job_id, "results": final_pages}
        out_path = local_dir / "final.json"
        with open(out_path, 'w') as f:
            json.dump(output, f)
            
        storage.upload_file(open(out_path, 'rb'), f"results/{job_id}/data.json", content_type="application/json")
        
        result_s3_key = f"results/{job_id}/data.json"
        
        # Cache the result for future uploads (Expire in 7 days)
        r.setex(f"file_hash:{file_hash}", 604800, result_s3_key)

        crud.update_job_status(
            db, 
            job_id, 
            "COMPLETED", 
            result_key=result_s3_key, # Variable from Phase 3 logic
            meta=meta
        )

        # --- OPTIMIZATION: Webhook ---
        if webhook_url:
            try:
                logger.info(f"[{job_id}] Dispatching Webhook to {webhook_url}")
                requests.post(webhook_url, json={
                    "event": "job.completed",
                    "job_id": job_id,
                    "status": "success",
                    "result_url": f"/jobs/{job_id}" # Or signed URL
                }, timeout=5)
            except Exception as e:
                logger.warning(f"Webhook failed: {e}")

        return {"status": "success", "result_key": result_s3_key}

    except SoftTimeLimitExceeded:
        logger.error("Task timed out!")
        crud.update_job_status(db, job_id, "FAILED", error=str(e))
        self.update_state(state='FAILURE', meta={'error': 'Processing timed out. File too large.'})
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise e
    finally:
        db.close() # CRITICAL: Close DB connection
        import shutil
        if local_dir.exists():
            shutil.rmtree(local_dir)