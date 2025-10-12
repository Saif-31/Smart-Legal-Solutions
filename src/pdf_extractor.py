import fitz  # PyMuPDF for PDF text extraction
import pytesseract  # Tesseract OCR
from PIL import Image
import logging
import os
import platform

def setup_tesseract():
    """Configure Tesseract based on environment"""
    try:
        if platform.system() == "Windows":
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        else:
            os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata"
    except Exception as e:
        logging.warning(f"Tesseract setup warning: {e}")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF and Tesseract OCR for image-based pages."""
    setup_tesseract()
    text_output = ""
    doc = None
    
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Try basic text extraction first
            text = page.get_text("text")
            if text.strip():
                text_output += text + "\n"
            else:
                try:
                    # Try OCR if available
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img, lang="srp")
                    text_output += ocr_text + "\n"
                except Exception as ocr_error:
                    logging.warning(f"OCR failed, using basic extraction: {ocr_error}")
                    text_output += text + "\n"
        
        return text_output.strip()
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        raise
    finally:
        try:
            if doc:
                doc.close()
            # Clean up temporary file
            if os.path.exists(pdf_path) and pdf_path.startswith("temp_"):
                os.remove(pdf_path)
        except Exception as cleanup_error:
            logging.warning(f"Error during cleanup: {cleanup_error}")
