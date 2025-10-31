"""
PDF text extraction with OCR support
"""
import os
import tempfile
from typing import Tuple
import PyPDF2
import ocrmypdf
from config import OCR_LANGUAGES, OCR_DPI, TEMP_DIR


class PDFProcessor:
    """Process PDFs with digital text extraction and OCR support"""

    def __init__(self):
        """Initialize PDF processor"""
        self.ocr_languages = OCR_LANGUAGES
        self.ocr_dpi = OCR_DPI

    def extract_text_digital(self, pdf_path: str) -> str:
        """
        Extract text from digital PDF (no OCR needed)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text = ""

        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                    except Exception as e:
                        print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                        continue

        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {e}")

        return text.strip()

    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """
        Extract text from scanned PDF using OCR
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        # Create temporary file for OCR output
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False, dir=TEMP_DIR) as tmp_file:
            output_path = tmp_file.name

        try:
            # Run OCRmyPDF
            print(f"Running OCR with languages: {self.ocr_languages}")
            ocrmypdf.ocr(
                pdf_path,
                output_path,
                language='+'.join(self.ocr_languages),
                deskew=True,
                rotate_pages=True,
                remove_background=False,
                clean=True,
                optimize=1,
                force_ocr=False,  # Only OCR pages that need it
                skip_text=False,
                redo_ocr=False,
                output_type='pdf',
                progress_bar=False
            )

            # Extract text from OCR'd PDF
            text = self.extract_text_digital(output_path)

            return text

        except Exception as e:
            raise Exception(f"OCR processing failed: {e}")

        finally:
            # Clean up temporary file
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass

    def detect_needs_ocr(self, pdf_path: str, sample_pages: int = 3) -> bool:
        """
        Detect if PDF needs OCR by checking if text is extractable
        
        Args:
            pdf_path: Path to PDF file
            sample_pages: Number of pages to sample
            
        Returns:
            True if OCR is needed, False otherwise
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)

                # Sample first few pages
                pages_to_check = min(sample_pages, total_pages)
                total_text = ""

                for i in range(pages_to_check):
                    try:
                        page_text = pdf_reader.pages[i].extract_text()
                        total_text += page_text
                    except:
                        continue

                # If we got very little text, probably needs OCR
                # Threshold: less than 50 characters per page on average
                avg_chars_per_page = len(total_text.strip()) / pages_to_check
                needs_ocr = avg_chars_per_page < 50

                print(f"Detected {avg_chars_per_page:.0f} chars/page. Needs OCR: {needs_ocr}")
                return needs_ocr

        except Exception as e:
            print(f"Error detecting OCR need: {e}")
            # If we can't determine, assume it needs OCR
            return True

    def process_pdf(self, pdf_path: str, force_ocr: bool = False) -> Tuple[str, bool]:
        """
        Process PDF and extract text (with OCR if needed)
        
        Args:
            pdf_path: Path to PDF file
            force_ocr: Force OCR even if text is extractable
            
        Returns:
            Tuple of (extracted_text, used_ocr)
        """
        # Detect if OCR is needed
        needs_ocr = force_ocr or self.detect_needs_ocr(pdf_path)

        if needs_ocr:
            print("Processing with OCR...")
            text = self.extract_text_with_ocr(pdf_path)
            used_ocr = True
        else:
            print("Processing as digital PDF...")
            text = self.extract_text_digital(pdf_path)
            used_ocr = False

        # Validate extracted text
        if len(text.strip()) < 50:
            raise Exception("Extracted text is too short. PDF may be corrupted or empty.")

        return text, used_ocr


def process_pdf_file(pdf_path: str, force_ocr: bool = False) -> Tuple[str, bool]:
    """
    Convenience function to process a PDF file
    
    Args:
        pdf_path: Path to PDF file
        force_ocr: Force OCR processing
        
    Returns:
        Tuple of (extracted_text, used_ocr)
    """
    processor = PDFProcessor()
    return processor.process_pdf(pdf_path, force_ocr=force_ocr)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_processor.py <pdf_file>")
        sys.exit(1)

    pdf_file = sys.argv[1]

    if not os.path.exists(pdf_file):
        print(f"Error: File not found: {pdf_file}")
        sys.exit(1)

    print(f"Processing: {pdf_file}")

    try:
        text, used_ocr = process_pdf_file(pdf_file)

        print(f"\n{'='*60}")
        print(f"Success! Used OCR: {used_ocr}")
        print(f"Extracted {len(text)} characters")
        print(f"{'='*60}")
        print(f"\nFirst 500 characters:")
        print(text[:500])

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
