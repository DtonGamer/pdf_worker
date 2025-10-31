"""
Text chunking utilities using token-based splitting
"""
import tiktoken
from typing import List, Dict
from config import CHUNK_SIZE, CHUNK_OVERLAP


class TextChunker:
    """Token-based text chunker for optimal embedding generation"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        """
        Initialize chunker with tiktoken encoder
        
        Args:
            chunk_size: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def chunk_text(self, text: str) -> List[Dict]:
        """
        Split text into overlapping chunks based on token count
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of dictionaries with 'text', 'index', and 'token_count'
        """
        # Encode text to tokens
        tokens = self.encoder.encode(text)

        chunks = []
        start_idx = 0
        chunk_index = 0

        while start_idx < len(tokens):
            # Get chunk of tokens
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode back to text
            chunk_text = self.encoder.decode(chunk_tokens)

            # Store chunk with metadata
            chunks.append({
                'text': chunk_text.strip(),
                'index': chunk_index,
                'token_count': len(chunk_tokens)
            })

            # Move to next chunk with overlap
            start_idx += self.chunk_size - self.overlap
            chunk_index += 1

        return chunks

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.encoder.encode(text))


def chunk_document(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Convenience function to chunk a document
    
    Args:
        text: Document text
        chunk_size: Maximum tokens per chunk
        overlap: Overlapping tokens
        
    Returns:
        List of chunk dictionaries
    """
    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
    return chunker.chunk_text(text)


if __name__ == "__main__":
    # Test chunking
    sample_text = """
    This is a sample document for testing the chunking functionality.
    It should be split into multiple chunks based on token count.
    Each chunk will have some overlap with the previous chunk to maintain context.
    """ * 100  # Repeat to create longer text

    chunks = chunk_document(sample_text, chunk_size=100, overlap=20)

    print(f"Created {len(chunks)} chunks")
    print(f"\nFirst chunk:")
    print(f"  Index: {chunks[0]['index']}")
    print(f"  Tokens: {chunks[0]['token_count']}")
    print(f"  Text preview: {chunks[0]['text'][:100]}...")
