import redis
import json
import os
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv('REDIS_URL')
QUEUE_NAME = 'pdf-processing'

# Connect to Redis
r = redis.from_url(REDIS_URL, decode_responses=True)

print("🔍 Checking Redis Queue...")
print(f"Queue: {QUEUE_NAME}\n")

# Check queue length
queue_length = r.llen(QUEUE_NAME)
print(f"Queue length: {queue_length}")

if queue_length > 0:
    # Peek at items without removing them
    items = r.lrange(QUEUE_NAME, 0, -1)

    print(f"\nItems in queue:")
    for i, item in enumerate(items):
        print(f"\n--- Item {i + 1} ---")
        print(f"Type: {type(item)}")
        print(f"Raw: {item[:200]}...")

        try:
            parsed = json.loads(item)
            print(f"Parsed type: {type(parsed)}")
            print(f"Parsed content: {parsed}")
        except Exception as e:
            print(f"Failed to parse: {e}")
else:
    print("\n✓ Queue is empty")

print("\n" + "="*60)
