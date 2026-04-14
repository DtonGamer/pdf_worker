"""
Bible seeder — processes a bible_seed job from the processing_queue.
Supports both URL downloads and Supabase Storage file downloads.
Downloads file, parses verses, generates embeddings, stores in knowledge_base.

Called from worker.py when job_data['job_type'] == 'bible_seed'.
Uses the same embeddings.py as the existing PDF pipeline (unchanged).
"""
import os
import json
import re
import tempfile
from typing import List, Dict, Any, Optional

from embeddings import generate_embeddings_batch
from config import BATCH_SIZE

# How many consecutive verses to group into one chunk
VERSES_PER_CHUNK = 8
# Insert this many KB rows per Supabase upsert call
INSERT_BATCH_SIZE = 100


def _update_bible_version(supabase, bible_version_id: str, status: str,
                           chunk_count: int = None, error: str = None):
    """Update bible_versions row status."""
    payload: Dict[str, Any] = {
        'status': status,
        'updated_at': 'now()'
    }
    if chunk_count is not None:
        payload['chunk_count'] = chunk_count
    if error:
        payload['error_message'] = error

    supabase.table('bible_versions').update(payload).eq('id', bible_version_id).execute()


def _download_from_storage(supabase, storage_path: str) -> bytes:
    """
    Download file from Supabase Storage bibles bucket.
    Returns raw file bytes.
    """
    print(f"  ⬇️  Downloading from Storage: {storage_path}...")
    
    # Use service role client for storage access
    response = supabase.storage.from_('bibles').download(storage_path)
    
    if isinstance(response, bytes):
        return response
    
    # Handle response object if needed
    if hasattr(response, 'content'):
        return response.content
    
    raise Exception(f"Unexpected response type from storage: {type(response)}")


def _parse_verses_from_json(data: Dict) -> List[Dict]:
    """
    Parse verses from nested JSON format (aruljohn/Bible-kjv style):
    { "Genesis": { "1": { "1": "In the beginning...", "2": "..." }, ... }, ... }
    """
    verses = []
    for book_name, chapters in data.items():
        if not isinstance(chapters, dict):
            continue
        for chapter_str, verse_map in chapters.items():
            if not isinstance(verse_map, dict):
                continue
            try:
                chapter_num = int(chapter_str)
            except ValueError:
                continue
            for verse_str, text in verse_map.items():
                try:
                    verse_num = int(verse_str)
                except ValueError:
                    continue
                if isinstance(text, str) and text.strip():
                    verses.append({
                        'book': book_name,
                        'chapter': chapter_num,
                        'verse': verse_num,
                        'text': text.strip()
                    })
    return verses


def _parse_verses_from_text(text: str) -> List[Dict]:
    """
    Parse verses from plain text format.
    Expected format: "Book Chapter:Verse Text" or similar patterns
    """
    verses = []
    
    # Pattern 1: "Genesis 1:1 In the beginning..."
    # Pattern 2: "1:1 In the beginning..." (with book header)
    # Pattern 3: "Genesis|1|1|In the beginning..." (pipe delimited)
    
    lines = text.split('\n')
    current_book = None
    
    # Common book names for detection
    book_names = [
        'Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy',
        'Joshua', 'Judges', 'Ruth', '1 Samuel', '2 Samuel',
        '1 Kings', '2 Kings', '1 Chronicles', '2 Chronicles',
        'Ezra', 'Nehemiah', 'Esther', 'Job', 'Psalms', 'Proverbs',
        'Ecclesiastes', 'Song of Solomon', 'Isaiah', 'Jeremiah',
        'Lamentations', 'Ezekiel', 'Daniel', 'Hosea', 'Joel',
        'Amos', 'Obadiah', 'Jonah', 'Micah', 'Nahum',
        'Habakkuk', 'Zephaniah', 'Haggai', 'Zechariah', 'Malachi',
        'Matthew', 'Mark', 'Luke', 'John', 'Acts',
        'Romans', '1 Corinthians', '2 Corinthians', 'Galatians',
        'Ephesians', 'Philippians', 'Colossians', '1 Thessalonians',
        '2 Thessalonians', '1 Timothy', '2 Timothy', 'Titus',
        'Philemon', 'Hebrews', 'James', '1 Peter', '2 Peter',
        '1 John', '2 John', '3 John', 'Jude', 'Revelation'
    ]
    
    # Try to detect format and parse
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if line starts with a book name (new book marker)
        for book in book_names:
            if line.startswith(book) and len(line) < len(book) + 5:
                current_book = book
                continue
        
        # Try "Book Chapter:Verse Text" format
        match = re.match(r'^([1-3]?\s?[A-Za-z]+)\s+(\d+):(\d+)\s+(.+)$', line)
        if match:
            book, chapter, verse, text = match.groups()
            verses.append({
                'book': book.strip(),
                'chapter': int(chapter),
                'verse': int(verse),
                'text': text.strip()
            })
            continue
        
        # Try "Chapter:Verse Text" format (requires current_book)
        if current_book:
            match = re.match(r'^(\d+):(\d+)\s+(.+)$', line)
            if match:
                chapter, verse, text = match.groups()
                verses.append({
                    'book': current_book,
                    'chapter': int(chapter),
                    'verse': int(verse),
                    'text': text.strip()
                })
                continue
        
        # Try pipe-delimited format: Book|Chapter|Verse|Text
        parts = line.split('|')
        if len(parts) == 4:
            book, chapter, verse, text = parts
            try:
                verses.append({
                    'book': book.strip(),
                    'chapter': int(chapter),
                    'verse': int(verse),
                    'text': text.strip()
                })
            except ValueError:
                pass
    
    return verses


def _parse_verses_from_pdf(file_bytes: bytes) -> List[Dict]:
    """
    Parse verses from PDF file bytes.
    Extracts text and then parses verses.
    """
    try:
        import PyPDF2
    except ImportError:
        raise Exception("PyPDF2 not installed. Run: pip install PyPDF2")
    
    text_parts = []
    pdf_file = io.BytesIO(file_bytes)
    reader = PyPDF2.PdfReader(pdf_file)
    
    for page in reader.pages:
        text_parts.append(page.extract_text())
    
    full_text = '\n'.join(text_parts)
    return _parse_verses_from_text(full_text)


def _load_and_parse_file(supabase, file_type: str, source: str, is_storage_path: bool = False) -> List[Dict]:
    """
    Load file from URL or Storage and parse verses based on file type.
    
    Args:
        file_type: 'json', 'txt', or 'pdf'
        source: URL string or storage path
        is_storage_path: True if source is storage path, False if URL
    """
    import requests
    import io
    
    # Download the file
    if is_storage_path:
        file_bytes = _download_from_storage(supabase, source)
    else:
        # URL download
        print(f"  ⬇️  Downloading from URL: {source}...")
        resp = requests.get(source, timeout=60)
        resp.raise_for_status()
        file_bytes = resp.content
        print(f"  ✓ Downloaded {len(file_bytes)} bytes")
    
    # Parse based on file type
    if file_type == 'json':
        # JSON format
        text = file_bytes.decode('utf-8')
        data = json.loads(text)
        verses = _parse_verses_from_json(data)
        
    elif file_type == 'txt':
        # Plain text format
        text = file_bytes.decode('utf-8')
        verses = _parse_verses_from_text(text)
        
    elif file_type == 'pdf':
        # PDF format
        verses = _parse_verses_from_pdf(file_bytes)
        
    else:
        raise Exception(f"Unsupported file type: {file_type}")
    
    # Sort canonically
    verses.sort(key=lambda v: (v['book'], v['chapter'], v['verse']))
    print(f"  ✓ Parsed {len(verses)} verses from {file_type.upper()}")
    
    return verses


def _group_into_chunks(verses: List[Dict], verses_per_chunk: int = VERSES_PER_CHUNK) -> List[Dict]:
    """
    Group consecutive verses into chunks, keeping within the same book.
    Each chunk carries:
      text, book, chapter, verse_start, verse_end, chunk_index
    """
    chunks = []
    chunk_index = 0
    i = 0

    while i < len(verses):
        current_book = verses[i]['book']
        group = []

        # Collect up to verses_per_chunk verses, stopping at book boundary
        while i < len(verses) and len(group) < verses_per_chunk:
            v = verses[i]
            if v['book'] != current_book:
                break
            group.append(v)
            i += 1

        # Build chunk text: "Book Chapter:Verse text\n..."
        lines = [
            f"{v['book']} {v['chapter']}:{v['verse']} {v['text']}"
            for v in group
        ]
        chunk_text = '\n'.join(lines)

        chunks.append({
            'text': chunk_text,
            'book': current_book,
            'chapter': group[0]['chapter'],
            'verse_start': group[0]['verse'],
            'verse_end': group[-1]['verse'],
            'chunk_index': chunk_index,
            'token_count': len(chunk_text.split())  # rough estimate
        })
        chunk_index += 1

    print(f"  ✓ Created {len(chunks)} chunks ({verses_per_chunk} verses each)")
    return chunks


def process_bible(supabase, job_data: Dict[str, Any], worker_id: str):
    """
    Main entry point called by worker.py for bible_seed jobs.

    Supports both URL and Supabase Storage file sources.

    job_data shape:
    {
      "job_type": "bible_seed",
      "bible_version_id": "<uuid>",
      "abbreviation": "KJV",
      "source_url": "https://...",           # Optional: external URL
      "source_storage_path": "bibles/...",    # Optional: storage path
      "source_file_type": "json|txt|pdf",    # Required with storage_path
      "triggered_by": "<user_uuid>"
    }
    """
    import io  # Import here for PDF parsing
    
    bible_version_id = job_data['bible_version_id']
    abbreviation = job_data.get('abbreviation', 'UNKNOWN')
    source_url = job_data.get('source_url')
    source_storage_path = job_data.get('source_storage_path')
    source_file_type = job_data.get('source_file_type', 'json')

    print(f"\n{'='*60}")
    print(f"[{worker_id}] 📖 Seeding Bible: {abbreviation} ({bible_version_id})")
    
    # Determine source
    if source_storage_path:
        print(f"[{worker_id}]    Source: Storage {source_storage_path} ({source_file_type})")
        is_storage = True
        source = source_storage_path
    elif source_url:
        print(f"[{worker_id}]    Source: URL {source_url}")
        is_storage = False
        source = source_url
    else:
        raise Exception("No source provided (need source_url or source_storage_path)")
    
    print(f"{'='*60}\n")

    try:
        # ── Step 1: Download and Parse ─────────────────────────────────────────
        _update_bible_version(supabase, bible_version_id, 'processing')
        
        verses = _load_and_parse_file(
            supabase, 
            source_file_type, 
            source, 
            is_storage_path=is_storage
        )
        
        total_verses = len(verses)

        # Update total_verses on the record
        supabase.table('bible_versions').update({
            'total_verses': total_verses,
            'updated_at': 'now()'
        }).eq('id', bible_version_id).execute()

        # ── Step 2: Group into chunks ─────────────────────────────────────────
        chunks = _group_into_chunks(verses, VERSES_PER_CHUNK)
        total_chunks = len(chunks)

        # ── Step 3: Delete any previous KB rows for this version (re-seed) ────
        print(f"[{worker_id}] 🗑️  Clearing previous chunks for {abbreviation}...")
        supabase.table('knowledge_base').delete().eq(
            'bible_version_id', bible_version_id
        ).execute()

        # ── Step 4: Generate embeddings in batches ────────────────────────────
        print(f"[{worker_id}] 🧠 Generating embeddings for {total_chunks} chunks...")
        all_embeddings = []
        total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in range(0, total_chunks, BATCH_SIZE):
            batch_chunks = chunks[batch_idx:batch_idx + BATCH_SIZE]
            batch_texts = [c['text'] for c in batch_chunks]

            progress_pct = int((batch_idx / total_chunks) * 100)
            print(f"[{worker_id}]   Embedding batch "
                  f"{batch_idx // BATCH_SIZE + 1}/{total_batches} "
                  f"({progress_pct}%)...")

            batch_embeddings = generate_embeddings_batch(batch_texts, batch_size=BATCH_SIZE)
            all_embeddings.extend(batch_embeddings)

        print(f"[{worker_id}] ✓ Generated {len(all_embeddings)} embeddings")

        # ── Step 5: Build and insert KB records ───────────────────────────────
        print(f"[{worker_id}] 💾 Inserting chunks into knowledge_base...")
        records = []
        for chunk, embedding in zip(chunks, all_embeddings):
            records.append({
                'bible_version_id': bible_version_id,
                'document_id': None,          # Not a user document
                'user_id': None,              # Shared, not per-user
                'content': chunk['text'],
                'embedding': embedding,
                'chunk_index': chunk['chunk_index'],
                'token_count': chunk['token_count'],
                'book': chunk['book'],
                'chapter': chunk['chapter'],
                'verse_start': chunk['verse_start'],
                'verse_end': chunk['verse_end'],
                'file_type': 'bible',
                'metadata': {
                    'bible_version_id': bible_version_id,
                    'abbreviation': abbreviation
                }
            })

        inserted = 0
        for i in range(0, len(records), INSERT_BATCH_SIZE):
            batch = records[i:i + INSERT_BATCH_SIZE]
            supabase.table('knowledge_base').insert(batch).execute()
            inserted += len(batch)
            print(f"[{worker_id}]   Inserted {inserted}/{total_chunks} chunks")

        # ── Step 6: Mark completed ────────────────────────────────────────────
        _update_bible_version(
            supabase, bible_version_id, 'completed',
            chunk_count=total_chunks
        )

        print(f"\n[{worker_id}] ✅ Bible seeding complete!")
        print(f"[{worker_id}]    Version:  {abbreviation}")
        print(f"[{worker_id}]    Verses:   {total_verses}")
        print(f"[{worker_id}]    Chunks:   {total_chunks}\n")

    except Exception as e:
        import traceback
        print(f"\n[{worker_id}] ❌ Bible seeding failed: {e}")
        print(traceback.format_exc())
        _update_bible_version(
            supabase, bible_version_id, 'failed',
            error=str(e)
        )
        raise
