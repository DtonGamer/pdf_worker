"""
Embedding generation using HuggingFace Inference API
No local model loaded — keeps memory well under 512MB on Render free tier
"""
import os
import time
import requests
from typing import List
from config import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS


HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
API_URL = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}"


def _call_api(texts: List[str], retries: int = 3) -> List[List[float]]:
    """
    Call HuggingFace Inference API to generate embeddings.
    Retries on model loading (503) with exponential backoff.
    """
    if not HUGGINGFACE_API_KEY:
        raise ValueError("HUGGINGFACE_API_KEY environment variable is not set")

    headers = {
        'Authorization': f'Bearer {HUGGINGFACE_API_KEY}',
        'Content-Type': 'application/json'
    }

    payload = {
        'inputs': texts,
        'options': {'wait_for_model': True}
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

            if response.status_code == 503:
                # Model is loading on HuggingFace side — wait and retry
                wait = 2 ** attempt
                print(f"⏳ Model loading on HuggingFace, retrying in {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            result = response.json()

            # API returns either [[...], [...]] or {"embeddings": [[...], [...]]}
            if isinstance(result, list):
                embeddings = result
            elif isinstance(result, dict) and 'embeddings' in result:
                embeddings = result['embeddings']
            else:
                raise ValueError(f"Unexpected API response format: {type(result)}")

            # Validate dimensions
            for emb in embeddings:
                if len(emb) != EMBEDDING_DIMENSIONS:
                    raise ValueError(
                        f"Dimension mismatch: expected {EMBEDDING_DIMENSIONS}, got {len(emb)}"
                    )

            return embeddings

        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                print(f"⚠️  API timeout, retrying ({attempt + 1}/{retries})...")
                time.sleep(2 ** attempt)
            else:
                raise Exception("HuggingFace API timed out after all retries")

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                print(f"⚠️  API error: {e}, retrying ({attempt + 1}/{retries})...")
                time.sleep(2 ** attempt)
            else:
                raise Exception(f"HuggingFace API failed: {e}")

    raise Exception("HuggingFace API failed after all retries")


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text.

    Args:
        text: Input text

    Returns:
        Embedding vector as list of floats
    """
    results = _call_api([text])
    return results[0]


def generate_embeddings_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Generate embeddings for a list of texts, batching API calls.

    Args:
        texts: List of input texts
        batch_size: Texts per API call (default 32 — safe for HuggingFace free tier)

    Returns:
        List of embedding vectors
    """
    all_embeddings = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        print(f"  Embedding batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} "
              f"({len(batch)} texts)...")
        embeddings = _call_api(batch)
        all_embeddings.extend(embeddings)

    return all_embeddings


if __name__ == "__main__":
    print("Testing HuggingFace API embedding generation...")
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"API URL: {API_URL}")

    test_texts = [
        "This is a test sentence.",
        "Another example for embedding generation.",
        "Machine learning is fascinating."
    ]

    print("\nTesting single embedding...")
    single = generate_embedding(test_texts[0])
    print(f"Dimension: {len(single)}")
    print(f"First 5 values: {single[:5]}")

    print("\nTesting batch embeddings...")
    batch = generate_embeddings_batch(test_texts)
    print(f"Batch shape: {len(batch)} x {len(batch[0])}")

    from numpy import dot, array
    from numpy.linalg import norm

    def cosine_similarity(a, b):
        a, b = array(a), array(b)
        return dot(a, b) / (norm(a) * norm(b))

    sim = cosine_similarity(batch[0], batch[1])
    print(f"\nSimilarity between first two texts: {sim:.4f}")
    print("\n✅ All tests passed!")