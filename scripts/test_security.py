import sys
import requests
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from blueprint_brain.src.db.session import SessionLocal
from blueprint_brain.src.security.service import SecurityService

def main():
    db = SessionLocal()
    
    # 1. Admin creates a key for "Acme Construction"
    print("Generating API Key for Acme Corp (Limit: 5 req/min)...")
    api_key = SecurityService.create_api_key(db, "Acme Corp", limit=5)
    print(f"API KEY: {api_key}")
    
    # 2. Simulate Traffic
    url = "http://localhost:8000/health" # Assuming auth protected or test endpoint
    headers = {"X-API-KEY": api_key}
    
    print("\nSimulating bursts...")
    for i in range(1, 10):
        # Note: In a real integration test, we'd hit the actual API. 
        # Here we assume the API is running locally on port 8000.
        try:
            resp = requests.get(url, headers=headers)
            print(f"Req {i}: Status {resp.status_code}")
            if resp.status_code == 429:
                print(">> Rate Limit Triggered Successfully!")
                break
        except Exception as e:
            print(f"API not running? {e}")
            break

if __name__ == "__main__":
    main()