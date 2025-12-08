import sys
import argparse
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from blueprint_brain.src.db.session import SessionLocal
from blueprint_brain.src.security.service import SecurityService

def main():
    parser = argparse.ArgumentParser(description="Create B2B API Key")
    parser.add_argument("name", help="Client Name (e.g., 'Skanska')")
    parser.add_argument("--limit", type=int, default=50, help="Rate limit per minute")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        print(f"Creating key for: {args.name}")
        raw_key = SecurityService.create_api_key(db, args.name, args.limit)
        print("\n" + "="*40)
        print(f"CLIENT: {args.name}")
        print(f"LIMIT:  {args.limit} req/min")
        print(f"API KEY: {raw_key}")
        print("="*40 + "\n")
        print("SAVE THIS KEY. It cannot be retrieved again.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    main()