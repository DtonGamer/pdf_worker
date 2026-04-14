"""
Main worker process for PDF document processing
"""
import os
import json
import time
import traceback
from typing import Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from supabase import create_client, Client
import mimetypes
import chardet

from config import (
    SUPABASE_URL, SUPABASE_SERVICE_KEY,
    TEMP_DIR, BATCH_SIZE,
    PORT, OCR_LANGUAGES, EMBEDDING_DIMENSIONS
)
from ocr_processor import process_pdf_file
from chunker import chunk_document
from embeddings import generate_embeddings_batch
from bible_seeder import process_bible


class PDFWorker:
    """Worker for processing PDF documents asynchronously"""

    def __init__(self):
        """Initialize worker with connections"""
        self.worker_id = os.getenv('WORKER_ID', f'worker-{int(time.time()) % 10000:04d}')
        print(f"🚀 Initializing PDF Worker [{self.worker_id}]...")

        # Legacy service role key — server-side only, never exposed to clients
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print(f"[{self.worker_id}] ✓ Connected to Supabase")

        self.poll_interval = int(os.getenv('POLL_INTERVAL', '5'))  # seconds
        print(f"[{self.worker_id}] ✓ Polling processing_queue every {self.poll_interval}s")
        print(f"[{self.worker_id}] ✓ Temp directory: {TEMP_DIR}")
        print(f"[{self.worker_id}] 🎯 Worker ready. Waiting for jobs...\n")

        # Initialize metrics
        self.jobs_processed = 0
        self.start_time = time.time()

    def update_status(self, document_id: str, status: str, message: str = None,
                     chunk_count: int = None, error: str = None):
        """Update document status in database"""
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
        """Download file directly from Supabase storage using service key"""
        if mime_type and mime_type.startswith('image/'):
            ext = mime_type.split('/')[-1]
            if ext == 'jpeg':
                ext = 'jpg'
        elif mime_type == 'application/pdf':
            ext = 'pdf'
        elif mime_type == 'text/plain':
            ext = 'txt'
        else:
            ext = 'bin'

        local_path = os.path.join(TEMP_DIR, f"{document_id}.{ext}")

        try:
            response = self.supabase.storage.from_('documents').download(storage_path)
            with open(local_path, 'wb') as f:
                f.write(response)
            return local_path

        except Exception as e:
            raise Exception(f"Failed to download file: {e}")

    def process_text_file(self, file_path: str) -> tuple[str, bool]:
        """
        Read a plain text file, auto-detecting encoding.

        Returns:
            Tuple of (extracted_text, used_ocr)
        """
        print(f"[{self.worker_id}] 📖 Reading text file...")

        with open(file_path, 'rb') as f:
            raw_bytes = f.read()

        # Detect encoding
        detected = chardet.detect(raw_bytes)
        encoding = detected.get('encoding') or 'utf-8'
        confidence = detected.get('confidence', 0)
        print(f"[{self.worker_id}] 🔍 Detected encoding: {encoding} (confidence: {confidence:.0%})")

        try:
            text = raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            print(f"[{self.worker_id}] ⚠️  Decoding failed with {encoding}, falling back to UTF-8")
            text = raw_bytes.decode('utf-8', errors='replace')

        text = text.strip()
        print(f"[{self.worker_id}] ✓ Read {len(text)} characters from text file")
        return text, False  # No OCR used

    def process_image_file(self, image_path: str) -> tuple[str, bool]:
        """Process an image file using OCR to extract text"""
        try:
            import pytesseract
            import cv2

            print(f"[{self.worker_id}] 🖼️  Loading image...")
            image = cv2.imread(image_path)

            print(f"[{self.worker_id}] 🖼️  Preprocessing image for OCR...")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            print(f"[{self.worker_id}] 🔍 Performing OCR on image...")
            text = pytesseract.image_to_string(thresh, lang='+'.join(OCR_LANGUAGES))
            text = text.strip()

            if len(text) < 10:
                text = pytesseract.image_to_string(image, lang='+'.join(OCR_LANGUAGES))
                text = text.strip()

            return text, True
        except Exception as e:
            raise Exception(f"Failed to process image: {e}")

    def process_document(self, job_data: Dict[str, Any]):
        """Process a single document (supports PDFs, images, and text files)"""
        document_id = job_data['document_id']
        storage_path = job_data['storage_path']
        needs_ocr = job_data.get('needs_ocr', False)
        user_id = job_data.get('user_id')

        # Prefer mime_type from job_data, fall back to guessing from path
        mime_type = job_data.get('mime_type') or mimetypes.guess_type(storage_path)[0]

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

            # Step 2: Download file directly via Supabase storage
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

            elif mime_type == 'text/plain':
                print(f"[{self.worker_id}] 📝 Processing as text file...")
                self.update_status(document_id, 'processing', 'Reading text file...')
                text, used_ocr = self.process_text_file(local_path)

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

                progress = int((batch_idx / len(chunks)) * 100)
                self.update_status(
                    document_id,
                    'processing',
                    f'Generating embeddings: {progress}%...'
                )

                batch_embeddings = generate_embeddings_batch(batch_texts, batch_size=BATCH_SIZE)
                all_embeddings.extend(batch_embeddings)

                print(f"[{self.worker_id}]   Batch {batch_idx // BATCH_SIZE + 1}/{total_batches} complete")

            print(f"[{self.worker_id}] ✓ Generated {len(all_embeddings)} embeddings")

            # Step 6: Store chunks + embeddings in database
            print(f"[{self.worker_id}] 💾 Storing embeddings in database...")
            self.update_status(document_id, 'processing', 'Storing embeddings...')

            records = []
            for chunk, embedding in zip(chunks, all_embeddings):
                records.append({
                    'document_id': document_id,
                    'content': chunk['text'],
                    'embedding': embedding,
                    'chunk_index': chunk['index'],
                    'token_count': chunk['token_count'],
                    'file_type': mime_type,
                    'user_id': user_id,
                    'metadata': {}
                })

            insert_batch_size = 100
            for i in range(0, len(records), insert_batch_size):
                batch = records[i:i + insert_batch_size]
                self.supabase.table('knowledge_base').upsert(
                    batch,
                    on_conflict='document_id,chunk_index'
                ).execute()
                print(f"[{self.worker_id}]   Upserted {min(i + insert_batch_size, len(records))}/{len(records)} chunks")

            print(f"[{self.worker_id}] ✓ Stored {len(records)} chunks in database")

            # Step 7: Update document metadata
            print(f"[{self.worker_id}] 📝 Updating document metadata...")
            embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
            self.supabase.table('documents').update({
                'metadata': {
                    'embedding_model': embedding_model,
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
            print(f"[{self.worker_id}]    Embedding model: {embedding_model}\n")

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

            # Re-raise so run() can mark the queue row as failed
            raise

        finally:
            if local_path and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                    print(f"[{self.worker_id}] 🗑️  Cleaned up temporary file")
                except Exception as e:
                    print(f"[{self.worker_id}] ⚠️  Failed to clean up temp file: {e}")

    def run(self):
        """Main worker loop - polls processing_queue table"""
        print(f"[{self.worker_id}] 🔄 Worker loop started\n")

        while True:
            try:
                result = (
                    self.supabase.table('processing_queue')
                    .select('*')
                    .eq('status', 'queued')
                    .order('created_at')
                    .limit(1)
                    .execute()
                )

                if not result.data:
                    time.sleep(self.poll_interval)
                    continue

                job_row = result.data[0]
                job_id = job_row['id']
                job_data = job_row['job_data']

                # Claim the job — double eq guards against two workers
                # picking up the same row simultaneously
                self.supabase.table('processing_queue').update({
                    'status': 'processing',
                    'updated_at': 'now()'
                }).eq('id', job_id).eq('status', 'queued').execute()

                # Process and finalize queue row
                # ── ROUTING: bible seed vs regular document ────────────────
                try:
                    job_type = job_data.get('job_type', 'document')

                    if job_type == 'bible_seed':
                        print(f"[{self.worker_id}] 📖 Routing to Bible seeder")
                        process_bible(self.supabase, job_data, self.worker_id)
                    else:
                        print(f"[{self.worker_id}] 📄 Routing to document processor")
                        self.process_document(job_data)

                    self.supabase.table('processing_queue').update({
                        'status': 'completed',
                        'processed_at': 'now()',
                        'updated_at': 'now()'
                    }).eq('id', job_id).execute()

                except Exception:
                    print(f"[{self.worker_id}] ❌ Marking queue job {job_id} as failed")
                    self.supabase.table('processing_queue').update({
                        'status': 'failed',
                        'processed_at': 'now()',
                        'updated_at': 'now()'
                    }).eq('id', job_id).execute()

            except KeyboardInterrupt:
                print(f"\n\n[{self.worker_id}] 🛑 Worker stopped by user")
                break

            except Exception as e:
                print(f"\n[{self.worker_id}] ❌ Unexpected error in worker loop: {e}")
                print(traceback.format_exc())
                time.sleep(5)


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Worker is running')

        elif self.path == '/metrics':
            import psutil
            process = psutil.Process()
            memory_usage_mb = process.memory_info().rss / 1024 / 1024

            worker = getattr(self.server, 'worker', None)

            queue_length = 0
            if worker:
                try:
                    res = (
                        worker.supabase.table('processing_queue')
                        .select('id', count='exact')
                        .eq('status', 'queued')
                        .execute()
                    )
                    queue_length = res.count or 0
                except Exception:
                    pass

            metrics = {
                'worker_id': getattr(worker, 'worker_id', 'unknown'),
                'jobs_processed': getattr(worker, 'jobs_processed', 0),
                'current_queue_length': queue_length,
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
        pass


def start_health_server(worker_instance):
    """Start a simple HTTP server for health checks"""
    server = HTTPServer(('0.0.0.0', PORT), HealthCheckHandler)
    server.worker = worker_instance
    server.start_time = time.time()
    print(f"[{worker_instance.worker_id}] ✓ Health check server listening on port {PORT}")
    server.serve_forever()


def main():
    """Entry point"""
    try:
        worker = PDFWorker()

        health_thread = threading.Thread(target=start_health_server, daemon=True, args=(worker,))
        health_thread.start()

        worker.run()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()
