"""
Configuration management for PDF processing worker
Fixed to handle environment variables with extra quotes (Zeabur issue)
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# HELPER FUNCTION TO STRIP QUOTES
# ============================================================================
def get_env_int(key: str, default: int) -> int:
    """
    Get integer environment variable, stripping any extra quotes
    Handles cases like BATCH_SIZE=32, BATCH_SIZE="32", BATCH_SIZE='"32"'
    """
    value = os.getenv(key)
    if value is None:
        return default

    # Strip whitespace and quotes (both single and double)
    value = value.strip().strip('"').strip("'")

    try:
        return int(value)
    except ValueError:
        print(f"Warning: Invalid value for {key}='{value}', using default={default}")
        return default


# ============================================================================
# SUPABASE CONFIGURATION
# ============================================================================
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

# ============================================================================
# REDIS CONFIGURATION
# ============================================================================
REDIS_URL = os.getenv('REDIS_URL')

if not REDIS_URL:
    raise ValueError("REDIS_URL must be set")

# Queue configuration for priority support
HIGH_PRIORITY_QUEUE = os.getenv('HIGH_PRIORITY_QUEUE', 'pdf-processing-high')
DEFAULT_QUEUE = os.getenv('DEFAULT_QUEUE', 'pdf-processing')
LOW_PRIORITY_QUEUE = os.getenv('LOW_PRIORITY_QUEUE', 'pdf-processing-low')

# For backward compatibility, maintain original queue name as default
QUEUE_NAME = os.getenv('QUEUE_NAME', DEFAULT_QUEUE)
QUEUE_TIMEOUT = 5  # ✨ CHANGED: Short timeout for health check responsiveness

# ============================================================================
# PORT CONFIGURATION (for health checks)
# ============================================================================
# Support both PORT (Railway/Render) and WEB_PORT (Zeabur)
PORT = int(os.getenv('PORT', os.getenv('WEB_PORT', '8080')))

# ============================================================================
# PROCESSING CONFIGURATION - FIXED FOR ALL PLATFORMS
# ============================================================================
# Ultra-small batch sizes for 512MB RAM limit (Render free tier)
BATCH_SIZE = get_env_int('BATCH_SIZE', 4)  # Reduced to 4 for extreme memory constraint
CHUNK_SIZE = get_env_int('CHUNK_SIZE', 500)  # tokens
CHUNK_OVERLAP = get_env_int('CHUNK_OVERLAP', 50)  # tokens

# ============================================================================
# OCR CONFIGURATION
# ============================================================================
OCR_LANGUAGES = os.getenv('OCR_LANGUAGES', 'eng').split(',')
OCR_DPI = get_env_int('OCR_DPI', 300)

# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================
# Use a better quality model while maintaining efficiency
# all-MiniLM-L6-v2: 23MB (better quality than L3-v2, same dimensions)
# paraphrase-MiniLM-L3-v2: 17MB (smallest, kept for reference)
EMBEDDING_MODEL = os.getenv(
    'EMBEDDING_MODEL', 
    'all-MiniLM-L6-v2'  # Better quality, still efficient
)
EMBEDDING_DIMENSIONS = 384

# ============================================================================
# TEMP FILE CONFIGURATION
# ============================================================================
TEMP_DIR = '/tmp/pdf-processing'
os.makedirs(TEMP_DIR, exist_ok=True)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# ============================================================================
# PRINT CONFIGURATION ON STARTUP
# ============================================================================
if __name__ == "__main__":
    print("Configuration loaded:")
    print(f"  PORT: {PORT}")
    print(f"  QUEUE_TIMEOUT: {QUEUE_TIMEOUT}s")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  CHUNK_SIZE: {CHUNK_SIZE}")
    print(f"  CHUNK_OVERLAP: {CHUNK_OVERLAP}")
    print(f"  EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    print(f"  OCR_LANGUAGES: {OCR_LANGUAGES}")
    print(f"  OCR_DPI: {OCR_DPI}")
