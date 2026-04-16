"""
Bible seeder — processes a bible_seed job from the bible_processing_queue.
Downloads file from Supabase Storage, parses verses, generates embeddings,
stores in knowledge_base.

Called from worker.py when a bible_processing_queue job is claimed.
Uses the same embeddings.py as the existing PDF pipeline (unchanged).
"""
import os
import json
import re
from typing import List, Dict, Any

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

    response = supabase.storage.from_('bibles').download(storage_path)

    if isinstance(response, bytes):
        return response

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
    Supports:
      - "Genesis 1:1 In the beginning..."
      - "1:1 In the beginning..."  (with book header line)
      - "Genesis|1|1|In the beginning..."
    """
    verses = []
    lines = text.split('\n')
    current_book = None

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

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for standalone book header line
        for book in book_names:
            if line.startswith(book) and len(line) < len(book) + 5:
                current_book = book
                continue

        # "Book Chapter:Verse Text"
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

        # "Chapter:Verse Text" (requires current_book context)
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

        # "Book|Chapter|Verse|Text"
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
    """Extract text from PDF bytes then parse as plain text."""
    import io
    try:
        import PyPDF2
    except ImportError:
        raise Exception("PyPDF2 not installed. Run: pip install PyPDF2")

    pdf_file = io.BytesIO(file_bytes)
    reader = PyPDF2.PdfReader(pdf_file)
    full_text = '\n'.join(page.extract_text() for page in reader.pages)
    return _parse_verses_from_text(full_text)


def _load_and_parse_file(supabase, file_type: str, storage_path: str) -> List[Dict]:
    """
    Download from Supabase Storage and parse verses based on file type.
    """
    file_bytes = _download_from_storage(supabase, storage_path)

    if file_type == 'json':
        data = json.loads(file_bytes.decode('utf-8'))
        verses = _parse_verses_from_json(data)

    elif file_type == 'txt':
        verses = _parse_verses_from_text(file_bytes.decode('utf-8'))

    elif file_type == 'pdf':
        verses = _parse_verses_from_pdf(file_bytes)

    else:
        raise Exception(f"Unsupported file type: {file_type}")

    verses.sort(key=lambda v: (v['book'], v['chapter'], v['verse']))
    print(f"  ✓ Parsed {len(verses)} verses from {file_type.upper()}")
    return verses


def _group_into_chunks(verses: List[Dict], verses_per_chunk: int = VERSES_PER_CHUNK) -> List[Dict]:
    """
    Group consecutive verses into chunks, keeping within the same book.
    """
    chunks = []
    chunk_index = 0
    i = 0

    while i < len(verses):
        current_book = verses[i]['book']
        group = []

        while i < len(verses) and len(group) < verses_per_chunk:
            v = verses[i]
            if v['book'] != current_book:
                break
            group.append(v)
            i += 1

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
            'token_count': len(chunk_text.split())
        })
        chunk_index += 1

    print(f"  ✓ Created {len(chunks)} chunks ({verses_per_chunk} verses each)")
    return chunks


def process_bible(supabase, job_data: Dict[str, Any], worker_id: str):
    """
    Main entry point called by worker.py for bible seed jobs.

    job_data shape:
    {
      "job_type": "bible_seed",
      "bible_version_id": "<uuid>",
      "abbreviation": "KJV",
      "source_storage_path": "bibles/user_id/filename.json",
      "source_file_type": "json|txt|pdf",
      "triggered_by": "<user_uuid>"
    }
    """
    bible_version_id = job_data['bible_version_id']
    abbreviation     = job_data.get('abbreviation', 'UNKNOWN')
    storage_path     = job_data.get('source_storage_path')
    file_type        = job_data.get('source_file_type', 'json')

    print(f"\n{'='*60}")
    print(f"[{worker_id}] 📖 Seeding Bible: {abbreviation} ({bible_version_id})")
    print(f"[{worker_id}]    Storage path: {storage_path}")
    print(f"[{worker_id}]    File type:    {file_type}")
    print(f"{'='*60}\n")

    if not storage_path:
        raise Exception("No source_storage_path provided in job_data")

    try:
        # ── Step 1: Download and parse ─────────────────────────────────────────
        _update_bible_version(supabase, bible_version_id, 'processing')

        verses = _load_and_parse_file(supabase, file_type, storage_path)
        total_verses = len(verses)

        supabase.table('bible_versions').update({
            'total_verses': total_verses,
            'updated_at': 'now()'
        }).eq('id', bible_version_id).execute()

        # ── Step 2: Group into chunks ──────────────────────────────────────────
        chunks = _group_into_chunks(verses, VERSES_PER_CHUNK)
        total_chunks = len(chunks)

        # ── Step 3: Clear previous chunks for this version (re-seed support) ──
        print(f"[{worker_id}] 🗑️  Clearing previous chunks for {abbreviation}...")
        supabase.table('knowledge_base').delete().eq(
            'bible_version_id', bible_version_id
        ).execute()

        # ── Step 4: Generate embeddings ────────────────────────────────────────
        print(f"[{worker_id}] 🧠 Generating embeddings for {total_chunks} chunks...")
        all_embeddings = []
        total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in range(0, total_chunks, BATCH_SIZE):
            batch_chunks = chunks[batch_idx:batch_idx + BATCH_SIZE]
            batch_texts  = [c['text'] for c in batch_chunks]

            progress_pct = int((batch_idx / total_chunks) * 100)
            print(f"[{worker_id}]   Embedding batch "
                  f"{batch_idx // BATCH_SIZE + 1}/{total_batches} "
                  f"({progress_pct}%)...")

            batch_embeddings = generate_embeddings_batch(batch_texts, batch_size=BATCH_SIZE)
            all_embeddings.extend(batch_embeddings)

        print(f"[{worker_id}] ✓ Generated {len(all_embeddings)} embeddings")

        # ── Step 5: Insert into knowledge_base ────────────────────────────────
        print(f"[{worker_id}] 💾 Inserting chunks into knowledge_base...")
        records = []
        for chunk, embedding in zip(chunks, all_embeddings):
            records.append({
                'bible_version_id': bible_version_id,
                'document_id':      None,   # not a user document
                'user_id':          None,   # shared, not per-user
                'content':          chunk['text'],
                'embedding':        embedding,
                'chunk_index':      chunk['chunk_index'],
                'token_count':      chunk['token_count'],
                'book':             chunk['book'],
                'chapter':          chunk['chapter'],
                'verse_start':      chunk['verse_start'],
                'verse_end':        chunk['verse_end'],
                'file_type':        'bible',
                'metadata': {
                    'bible_version_id': bible_version_id,
                    'abbreviation':     abbreviation
                }
            })

        inserted = 0
        for i in range(0, len(records), INSERT_BATCH_SIZE):
            batch = records[i:i + INSERT_BATCH_SIZE]
            supabase.table('knowledge_base').insert(batch).execute()
            inserted += len(batch)
            print(f"[{worker_id}]   Inserted {inserted}/{total_chunks} chunks")

        # ── Step 6: Mark completed ─────────────────────────────────────────────
        _update_bible_version(
            supabase, bible_version_id, 'completed',
            chunk_count=total_chunks
        )

        print(f"\n[{worker_id}] ✅ Bible seeding complete!")
        print(f"[{worker_id}]    Version: {abbreviation}")
        print(f"[{worker_id}]    Verses:  {total_verses}")
        print(f"[{worker_id}]    Chunks:  {total_chunks}\n")

    except Exception as e:
        import traceback
        print(f"\n[{worker_id}] ❌ Bible seeding failed: {e}")
        print(traceback.format_exc())
        _update_bible_version(
            supabase, bible_version_id, 'failed',
            error=str(e)
        )
        raise
