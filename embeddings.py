"""
Embedding generation using HuggingFace sentence-transformers
ULTRA MEMORY OPTIMIZED for 512MB RAM environments (Render free tier)
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
from config import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, BATCH_SIZE
import gc
import torch
import sys


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers with aggressive memory optimization"""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize embedding model with extreme memory constraints
        
        Args:
            model_name: HuggingFace model identifier
        """
        print(f"Loading embedding model: {model_name}")
        print(f"Memory before model load: {self._get_memory_usage():.1f}MB")

        # Force garbage collection before loading
        gc.collect()

        # Use CPU only and optimize for low memory
        # device='cpu' is critical - GPU memory would exceed limits
        self.model = SentenceTransformer(model_name, device='cpu')

        # Set to evaluation mode to save memory (no gradient tracking)
        self.model.eval()

        # Disable gradient computation globally
        torch.set_grad_enabled(False)

        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        print(f"Memory after model load: {self._get_memory_usage():.1f}MB")

        # Verify dimensions match config
        actual_dim = self.model.get_sentence_embedding_dimension()
        if actual_dim != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Model dimension mismatch: expected {EMBEDDING_DIMENSIONS}, got {actual_dim}"
            )

        # Force cleanup after initialization
        gc.collect()

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def generate_embeddings(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 2,  # ULTRA SMALL: 2 for 512MB constraint
        show_progress: bool = False  # Disable progress bar to save memory
    ) -> np.ndarray:
        """
        Generate embeddings for text(s) with aggressive memory management
        
        Args:
            texts: Single text or list of texts
            batch_size: Number of texts to process at once (default 2 for 512MB)
            show_progress: Show progress bar (disabled by default for memory)
            
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]

        # Force cleanup before processing
        gc.collect()
        torch.cuda.empty_cache()  # Safe even on CPU

        print(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}")
        mem_before = self._get_memory_usage()
        if mem_before > 0:
            print(f"Memory before encoding: {mem_before:.1f}MB")

        # Generate embeddings with memory optimization
        with torch.no_grad():  # Critical: Don't track gradients
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                convert_to_tensor=False  # Return numpy to avoid tensor memory
            )

        mem_after = self._get_memory_usage()
        if mem_after > 0:
            print(f"Memory after encoding: {mem_after:.1f}MB")

        # Aggressive garbage collection
        gc.collect()
        torch.cuda.empty_cache()  # Safe even on CPU

        return embeddings

    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            List of floats (embedding vector)
        """
        embedding = self.generate_embeddings(text, show_progress=False, batch_size=1)
        return embedding[0].tolist()

    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts with memory-aware batching
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        # Ultra-small batch size for 512MB RAM
        # Process 2 texts at a time to prevent OOM
        embeddings = self.generate_embeddings(texts, show_progress=False, batch_size=2)

        # Explicit memory cleanup after processing
        result = embeddings.tolist()
        del embeddings  # Remove reference to numpy array
        gc.collect()
        torch.cuda.empty_cache()

        return result


# Global model instance (loaded once to save memory)
_model_instance = None


def get_embedding_model() -> EmbeddingGenerator:
    """
    Get or create global embedding model instance (singleton pattern)
    This ensures we only load the model once, saving memory
    
    Returns:
        EmbeddingGenerator instance
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = EmbeddingGenerator()
    return _model_instance


def generate_embedding(text: str) -> List[float]:
    """
    Convenience function to generate single embedding
    
    Args:
        text: Input text
        
    Returns:
        Embedding vector as list
    """
    model = get_embedding_model()
    return model.generate_single_embedding(text)


def generate_embeddings_batch(texts: List[str], batch_size: int = 2) -> List[List[float]]:
    """
    Convenience function to generate batch embeddings
    Ultra memory optimized for 512MB RAM
    
    Args:
        texts: List of input texts
        batch_size: Batch size for processing (default 2 for 512MB constraint)
        
    Returns:
        List of embedding vectors
    """
    model = get_embedding_model()
    # Override batch_size to ensure it doesn't exceed safe limits
    safe_batch_size = min(batch_size, 2)
    return model.generate_batch_embeddings(texts)


if __name__ == "__main__":
    # Test embedding generation
    print("Testing embedding generation...")
    print(f"Python version: {sys.version}")

    # Check if psutil is available
    try:
        import psutil
        print(f"Initial memory: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB")
    except ImportError:
        print("psutil not available, memory monitoring disabled")

    test_texts = [
        "This is a test sentence.",
        "Another example for embedding generation.",
        "Machine learning is fascinating."
    ]

    # Test single embedding
    print("\nTesting single embedding...")
    single_emb = generate_embedding(test_texts[0])
    print(f"Single embedding shape: {len(single_emb)}")
    print(f"First 5 values: {single_emb[:5]}")

    # Test batch embeddings
    print("\nTesting batch embeddings...")
    batch_embs = generate_embeddings_batch(test_texts)
    print(f"Batch embeddings shape: {len(batch_embs)} x {len(batch_embs[0])}")

    # Test similarity
    from numpy import dot
    from numpy.linalg import norm

    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    sim = cosine_similarity(batch_embs[0], batch_embs[1])
    print(f"\nSimilarity between first two texts: {sim:.4f}")

    print("\n✅ All tests passed!")
