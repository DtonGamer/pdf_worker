"""
Main worker process for PDF document processing
"""
import os
import json
import time
import traceback
from typing import Dict, Any
import redis
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from supabase import create_client, Client
import mimetypes

from config import (
    SUPABASE_URL, SUPABASE_SERVICE_KEY, REDIS_URL, 
    QUEUE_NAME, QUEUE_TIMEOUT, TEMP_DIR, BATCH_SIZE,
    PORT, OCR_LANGUAGES, EMBEDDING_DIMENSIONS
)
from ocr_processor import process_pdf_file
from chunker import chunk_document
from embeddings import generate_embeddings_batch


class PDFWorker:
    """Worker for processing PDF documents asynchronously"""

    def __init__(self):
        """Initialize worker with connections"""
        self.worker_id = os.getenv('WORKER_ID', f'worker-{int(time.time()) % 10000:04d}')
        print(f"🚀 Initializing PDF Worker [{self.worker_id}]...")

        # Initialize Supabase client
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print(f"[{self.worker_id}] ✓ Connected to Supabase")

        # Initialize Redis client
        self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        self.redis_client.ping()
        print(f"[{self.worker_id}] ✓ Connected to Redis")

        print(f"[{self.worker_id}] ✓ Listening to queue: {QUEUE_NAME}")
        print(f"[{self.worker_id}] ✓ Temp directory: {TEMP_DIR}")
        print(f"[{self.worker_id}] 🎯 Worker ready. Waiting for jobs...\n")

        # Initialize metrics
        self.jobs_processed = 0
        self.start_time = time.time()

    def update_status(self, document_id: str, status: str, message: str = None, 
                     chunk_count: int = None, error: str = None):
        """
        Update document status in database
        
        Args:
            document_id: Document UUID
            status: New status
            message: Status message
            chunk_count: Number of chunks processed
            error: Error message if failed
        """
        try:
            update_data = {
                'status': status,
                'updated_at': 'now()'
            }

            if message:
                update_data['status_message'] = message

            if chunk_count is not None:
                update_data['chunk_count'] = chunk_count

            if error:
                update_data['error_message'] = error

            if status == 'processing':
                update_data['processing_started_at'] = 'now()'
            elif status in ['completed', 'failed', 'partial']:
                update_data['processing_completed_at'] = 'now()'

            self.supabase.table('documents').update(update_data).eq('id', document_id).execute()

        except Exception as e:
            print(f"⚠️  Failed to update status: {e}")

    def download_file(self, storage_path: str, document_id: str, mime_type: str) -> str:
        """
        Download file from Supabase storage based on file type
        
        Args:
            storage_path: Path in storage bucket
            document_id: Document UUID
            mime_type: MIME type of the file
            
        Returns:
            Local file path
        """
        # Determine file extension based on MIME type
        if mime_type and mime_type.startswith('image/'):
            ext = mime_type.split('/')[-1]
            if ext == 'jpeg':
                ext = 'jpg'
        elif mime_type == 'application/pdf':
            ext = 'pdf'
        else:
            ext = 'bin'  # Default extension

        local_path = os.path.join(TEMP_DIR, f"{document_id}.{ext}")

        try:
            # Download file from storage
            response = self.supabase.storage.from_('documents').download(storage_path)

            # Save to local file
            with open(local_path, 'wb') as f:
                f.write(response)

            return local_path

        except Exception as e:
            raise Exception(f"Failed to download file: {e}")

    def process_image_file(self, image_path: str) -> tuple[str, bool]:
        """
        Process an image file using OCR to extract text
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (extracted_text, used_ocr)
        """
        try:
            from PIL import Image
            import pytesseract
            import cv2
            import numpy as np

            print(f"[{self.worker_id}] 🖼️  Loading image...")

            # Load image using OpenCV for preprocessing
            image = cv2.imread(image_path)

            # Preprocess image for better OCR
            print(f"[{self.worker_id}] 🖼️  Preprocessing image for OCR...")

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Use adaptive thresholding for better text detection
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            # Perform OCR using pytesseract
            print(f"[{self.worker_id}] 🔍 Performing OCR on image...")
            text = pytesseract.image_to_string(thresh, lang='+'.join(OCR_LANGUAGES))

            # Clean up text
            text = text.strip()

            # If pytesseract didn't work well, try with original image
            if len(text) < 10:
                text = pytesseract.image_to_string(image, lang='+'.join(OCR_LANGUAGES))
                text = text.strip()

            return text, True  # Always return True for OCR used with images
        except Exception as e:
            raise Exception(f"Failed to process image: {e}")

    def process_document(self, job_data: Dict[str, Any]):
        """
        Process a single document (supports PDFs and images)
        
        Args:
            job_data: Job information from queue
        """
        document_id = job_data['document_id']
        storage_path = job_data['storage_path']
        needs_ocr = job_data.get('needs_ocr', False)
        user_id = job_data.get('user_id')

        # Detect file type
        mime_type, _ = mimetypes.guess_type(storage_path)

        print(f"\n{'='*60}")
        print(f"[{self.worker_id}] 📄 Processing document: {document_id}")
        print(f"[{self.worker_id}]    Storage path: {storage_path}")
        print(f"[{self.worker_id}]    File type: {mime_type}")
        print(f"[{self.worker_id}]    Needs OCR: {needs_ocr}")
        print(f"[{self.worker_id}]    User ID: {user_id}")
        print(f"{'='*60}\n")

        local_path = None

        try:
            # Step 1: Update status to processing
            self.update_status(document_id, 'processing', 'Starting processing...')

            # Step 2: Download file
            print(f"[{self.worker_id}] ⬇️  Downloading file...")
            self.update_status(document_id, 'processing', 'Downloading document...')
            local_path = self.download_file(storage_path, document_id, mime_type)
            print(f"[{self.worker_id}] ✓ Downloaded to: {local_path}")

            # Step 3: Process based on file type
            if mime_type and mime_type.startswith('image/'):
                print(f"[{self.worker_id}] 🖼️  Processing as image...")
                self.update_status(document_id, 'processing', 'Processing image file...')
                text, used_ocr = self.process_image_file(local_path)
            elif mime_type == 'application/pdf':
                print(f"[{self.worker_id}] 📄 Processing as PDF...")
                if needs_ocr:
                    print(f"[{self.worker_id}] 🔍 Extracting text with OCR...")
                    self.update_status(document_id, 'processing', 'Running OCR (this may take a while)...')
                else:
                    print(f"[{self.worker_id}] 📝 Extracting text from digital PDF...")
                    self.update_status(document_id, 'processing', 'Extracting text...')

                text, used_ocr = process_pdf_file(local_path, force_ocr=needs_ocr)
            else:
                raise ValueError(f"Unsupported file type: {mime_type}")

            print(f"[{self.worker_id}] ✓ Extracted {len(text)} characters (OCR: {used_ocr})")

            # Step 4: Chunk text
            print(f"[{self.worker_id}] ✂️  Chunking text...")
            self.update_status(document_id, 'processing', 'Creating text chunks...')
            chunks = chunk_document(text)
            print(f"[{self.worker_id}] ✓ Created {len(chunks)} chunks")

            # Step 5: Generate embeddings in batches
            print(f"[{self.worker_id}] 🧠 Generating embeddings...")
            all_embeddings = []
            total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

            for batch_idx in range(0, len(chunks), BATCH_SIZE):
                batch_chunks = chunks[batch_idx:batch_idx + BATCH_SIZE]
                batch_texts = [chunk['text'] for chunk in batch_chunks]

                # Update progress
                progress = int((batch_idx / len(chunks)) * 100)
                self.update_status(
                    document_id, 
                    'processing', 
                    f'Generating embeddings: {progress}%...'
                )

                # Generate embeddings for batch
                batch_embeddings = generate_embeddings_batch(batch_texts, batch_size=BATCH_SIZE)
                all_embeddings.extend(batch_embeddings)

                print(f"[{self.worker_id}]   Batch {batch_idx // BATCH_SIZE + 1}/{total_batches} complete")

            print(f"[{self.worker_id}] ✓ Generated {len(all_embeddings)} embeddings")

            # Step 6: Store in database
            print(f"[{self.worker_id}] 💾 Storing embeddings in database...")
            self.update_status(document_id, 'processing', 'Storing embeddings...')

            # Prepare records for insertion
            records = []
            for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                records.append({
                    'document_id': document_id,
                    'content': chunk['text'],
                    'embedding': embedding,
                    'chunk_index': chunk['index'],
                    'token_count': chunk['token_count'],
                    'file_type': mime_type,
                    'user_id': user_id
                })

            # Insert in batches of 100 using upsert to handle re-processing scenarios
            insert_batch_size = 100
            for i in range(0, len(records), insert_batch_size):
                batch = records[i:i + insert_batch_size]
                # Using upsert with conflict resolution on document_id and chunk_index
                self.supabase.table('knowledge_base').upsert(batch, on_conflict='document_id,chunk_index').execute()
                print(f"[{self.worker_id}]   Upserted {min(i + insert_batch_size, len(records))}/{len(records)} chunks")

            print(f"[{self.worker_id}] ✓ Stored {len(records)} chunks in database")

            # Step 7: Update metadata to track embedding model used
            print(f"[{self.worker_id}] 📝 Updating document metadata...")
            self.supabase.table('documents').update({
                'metadata': {
                    'embedding_model': os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5'),
                    'embedding_dimensions': EMBEDDING_DIMENSIONS,
                    'processed_at': time.time()
                }
            }).eq('id', document_id).execute()

            # Step 8: Mark as completed
            self.update_status(
                document_id, 
                'completed', 
                f'Successfully processed {len(chunks)} chunks',
                chunk_count=len(chunks)
            )

            print(f"\n[{self.worker_id}] ✅ Successfully processed document {document_id}")
            print(f"[{self.worker_id}]    Total chunks: {len(chunks)}")
            print(f"[{self.worker_id}]    Used OCR: {used_ocr}")
            print(f"[{self.worker_id}]    Embedding model: {os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')}\n")

            # Update metrics
            self.jobs_processed += 1

        except Exception as e:
            error_msg = str(e)
            print(f"\n[{self.worker_id}] ❌ Error processing document {document_id}: {error_msg}")
            print(traceback.format_exc())

            self.update_status(
                document_id,
                'failed',
                'Processing failed',
                error=error_msg
            )

        finally:
            # Clean up temporary file
            if local_path and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                    print(f"[{self.worker_id}] 🗑️  Cleaned up temporary file")
                except Exception as e:
                    print(f"[{self.worker_id}] ⚠️  Failed to clean up temp file: {e}")

    def run(self):
        """Main worker loop"""
        print(f"[{self.worker_id}] 🔄 Worker loop started\n")

        while True:
            try:
                # Use short timeout for health check responsiveness
                result = self.redis_client.blpop(QUEUE_NAME, timeout=QUEUE_TIMEOUT)

                if result is None:
                    # Timeout - no jobs available
                    # This is normal, just continue waiting
                    continue

                # Parse job data
                queue_name, job_json = result
                job_data = json.loads(job_json)

                # Handle case where job_data is a list containing one dictionary
                if isinstance(job_data, list) and len(job_data) > 0:
                    job_data = job_data[0]

                # Process the document
                self.process_document(job_data)

            except KeyboardInterrupt:
                print(f"\n\n[{self.worker_id}] 🛑 Worker stopped by user")
                break

            except Exception as e:
                print(f"\n[{self.worker_id}] ❌ Unexpected error in worker loop: {e}")
                print(traceback.format_exc())
                # Wait a bit before continuing
                time.sleep(5)


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Worker is running')
        elif self.path == '/metrics':
            # Return processing metrics
            import psutil
            process = psutil.Process()
            memory_usage_mb = process.memory_info().rss / 1024 / 1024

            # Get worker instance to access metrics
            worker = getattr(self.server, 'worker', None)

            metrics = {
                'worker_id': getattr(worker, 'worker_id', 'unknown'),
                'jobs_processed': getattr(worker, 'jobs_processed', 0),
                'current_queue_length': getattr(worker, 'redis_client', redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'), decode_responses=True)).llen(os.getenv('QUEUE_NAME', 'pdf-processing')) if worker else 0,
                'uptime_seconds': time.time() - getattr(worker, 'start_time', time.time()),
                'memory_usage_mb': round(memory_usage_mb, 2),
                'timestamp': time.time()
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(metrics).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress HTTP logs
        pass


def start_health_server(worker_instance):
    """Start a simple HTTP server for health checks"""
    server = HTTPServer(('0.0.0.0', PORT), HealthCheckHandler)
    # Attach worker instance to server for metrics access
    server.worker = worker_instance
    server.start_time = time.time()
    print(f"[{worker_instance.worker_id}] ✓ Health check server listening on port {PORT}")
    server.serve_forever()


def main():
    """Entry point"""
    try:
        # Start worker first
        worker = PDFWorker()

        # Start health check server in background thread, passing worker instance
        health_thread = threading.Thread(target=start_health_server, daemon=True, args=(worker,))
        health_thread.start()

        # Start worker loop
        worker.run()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()
