import boto3
import logging
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from blueprint_brain.config.settings import settings

logger = logging.getLogger(__name__)

class StorageService:
    """
    Wrapper for S3-compatible Object Storage.
    """
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            endpoint_url=settings.S3_ENDPOINT,
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket = settings.S3_BUCKET_NAME
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except ClientError:
            logger.info(f"Bucket {self.bucket} not found. Creating...")
            try:
                self.s3.create_bucket(Bucket=self.bucket)
                self._set_lifecycle_rules()
            except Exception as e:
                logger.error(f"Failed to create bucket: {e}")


    def _set_lifecycle_rules(self):
        """
        Configures MinIO/S3 to auto-delete temp files.
        - 'uploads/' folder: Delete after 1 day
        - 'results/' folder: Keep for 30 days (or move to Glacier)
        """
        try:
            self.s3.put_bucket_lifecycle_configuration(
                Bucket=self.bucket,
                LifecycleConfiguration={
                    'Rules': [
                        {
                            'ID': 'ExpireRawUploads',
                            'Prefix': 'uploads/',
                            'Status': 'Enabled',
                            'Expiration': {'Days': 1} # Raw PDF gone in 24h
                        },
                        {
                            'ID': 'ExpireTempResults',
                            'Prefix': 'results/',
                            'Status': 'Enabled',
                            'Expiration': {'Days': 30} # JSON results gone in 30 days
                        }
                    ]
                }
            )
            logger.info("Lifecycle rules applied to bucket.")
        except Exception as e:
            logger.warning(f"Could not set lifecycle rules: {e}")

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ClientError)
    )
    def upload_file(self, file_obj, object_name: str) -> bool:
        """Uploads a file-like object to S3."""
        try:
            self.s3.upload_fileobj(file_obj, self.bucket, object_name)
            logger.info(f"Uploaded {object_name} to S3.")
            return True
        except ClientError as e:
            logger.error(f"Upload failed: {e}")
            return False
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def download_file(self, object_name: str, local_path: str) -> bool:
        """Downloads file from S3 to local path."""
        try:
            self.s3.download_file(self.bucket, object_name, local_path)
            return True
        except ClientError as e:
            logger.error(f"Download failed: {e}")
            return False
            
    def generate_presigned_url(self, object_name: str, expiration=3600) -> str:
        """Generates a temporary URL for downloading results."""
        try:
            response = self.s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': object_name},
                ExpiresIn=expiration
            )
            return response
        except ClientError as e:
            logger.error(f"Presigned URL generation failed: {e}")
            return ""
        
    def generate_presigned_upload(self, object_name: str, expiration=3600):
        """
        Generates a Presigned POST to allow client to upload directly to S3/MinIO.
        Returns: {'url': '...', 'fields': {...}}
        """
        try:
            response = self.s3.generate_presigned_post(
                Bucket=self.bucket,
                Key=object_name,
                Fields=None,
                Conditions=[
                    ["content-length-range", 0, 524288000]  # Max 500MB
                ],
                ExpiresIn=expiration
            )
            return response
        except ClientError as e:
            logger.error(f"Presigned upload generation failed: {e}")
            raise e