"""
Bible seeder — processes a bible_seed job from the processing_queue.
Downloads KJV JSON, chunks verses, generates embeddings, stores in knowledge_base.

Called from worker.py when job_data['job_type'] == 'bible_seed'.
Uses the same embeddings.py and chunker.py as the existing PDF pipeline.
"""
import os
import json
import time
import requests
from typing import List, Dict, Any

from embeddings import generate_embeddings_batch
from config import BATCH_SIZE, EMBEDDING_DIMENSIONS

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


def _download_bible_json(source_url: str) -> Dict:
    """
    Download Bible JSON from source_url.
    Expected format (aruljohn/Bible-kjv):
    {
      "Genesis": { "1": { "1": "In the beginning...", "2": "..." }, ... },
      ...
    }
    """
    print(f"  ⬇️  Downloading Bible JSON from {source_url}...")
    resp = requests.get(source_url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    print(f"  ✓ Downloaded — {len(data)} books found")
    return data


def _build_verse_list(bible_data: Dict) -> List[Dict]:
    """
    Flatten the nested JSON into a list of verse dicts:
    { book, chapter (int), verse (int), text }
    """
    verses = []
    for book_name, chapters in bible_data.items():
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
    # Sort canonically
    verses.sort(key=lambda v: (v['book'], v['chapter'], v['verse']))
    print(f"  ✓ Flattened {len(verses)} verses")
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

    job_data shape:
    {
      "job_type": "bible_seed",
      "bible_version_id": "<uuid>",
      "abbreviation": "KJV",
      "source_url": "https://...",
      "triggered_by": "<user_uuid>"
    }
    """
    bible_version_id = job_data['bible_version_id']
    abbreviation = job_data.get('abbreviation', 'UNKNOWN')
    source_url = job_data.get('source_url')

    print(f"\n{'='*60}")
    print(f"[{worker_id}] 📖 Seeding Bible: {abbreviation} ({bible_version_id})")
    print(f"[{worker_id}]    Source: {source_url}")
    print(f"{'='*60}\n")

    try:
        # ── Step 1: Download ──────────────────────────────────────────────────
        _update_bible_version(supabase, bible_version_id, 'processing')
        bible_data = _download_bible_json(source_url)

        # ── Step 2: Flatten to verse list ─────────────────────────────────────
        verses = _build_verse_list(bible_data)
        total_verses = len(verses)

        # Update total_verses on the record
        supabase.table('bible_versions').update({
            'total_verses': total_verses,
            'updated_at': 'now()'
        }).eq('id', bible_version_id).execute()

        # ── Step 3: Group into chunks ─────────────────────────────────────────
        chunks = _group_into_chunks(verses, VERSES_PER_CHUNK)
        total_chunks = len(chunks)

        # ── Step 4: Delete any previous KB rows for this version (re-seed) ────
        print(f"[{worker_id}] 🗑️  Clearing previous chunks for {abbreviation}...")
        supabase.table('knowledge_base').delete().eq(
            'bible_version_id', bible_version_id
        ).execute()

        # ── Step 5: Generate embeddings in batches ────────────────────────────
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

        # ── Step 6: Build and insert KB records ───────────────────────────────
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

        # ── Step 7: Mark completed ────────────────────────────────────────────
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
