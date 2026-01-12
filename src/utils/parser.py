import os
import re
import PyPDF2
from docx import Document

class ResumeParser:
    """
    Responsible for converting unstructured Binary Data (PDF/Docx)
    into Clean Text Stream.
    """

    @staticmethod
    def extract_text_from_pdf(pdf_path) -> str:
        """
        Extracts raw text from PDF using a layer-based approach.
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file) 
                for page in reader.pages:
                    # extract_text() handles character encoding maps
                    content = page.extract_text()
                    if content:
                        text += content + "\n"
        except Exception as e:
            print(f"Error parsing PDF: {e}")
        return text

    @staticmethod
    def extract_text_from_docx(docx_path: str) -> str:
        """
        Extracts text from Word documents.
        """
        try:
            doc = Document(docx_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"Error parsing DOCX: {e}")
            return ""
        
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Pre-processing step to improve Signal-to-Noise Ratio.
        """
        # 1. Normalize Ehitespace: Mathematical models (like Word2Vec)
        # see 'Data  Science' and 'Data Science' as different inputs.
        text = re.sub(r'\s+', ' ', text)

        # 2. Handle Special Characters: Removes non-ascii symbols
        # that dont contribute to semantic meaning.
        text = re.sub(r'[^x00-\x7f]', r' ', text)

        return text.strip()
            

               
        
    