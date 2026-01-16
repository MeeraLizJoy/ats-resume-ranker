import os
import pdfplumber # Upgraded from PyPDF2 for better accuracy
from docx import Document

class ResumeParser:
    """
    Responsible for converting unstructured Binary Data (PDF/Docx/Txt)
    into a Raw Text Stream.
    """

    def extract_text(self, file_path: str) -> str:
        """
        Universal dispatcher that selects the correct extraction method.
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif ext == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            return f"Error: Unsupported file format {ext}"

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    content = page.extract_text()
                    if content:
                        text += content + "\n"
        except Exception as e:
            return f"Error parsing PDF: {e}"
        return text

    def extract_text_from_docx(self, docx_path: str) -> str:
        try:
            doc = Document(docx_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            return f"Error parsing DOCX: {e}"

    def extract_text_from_txt(self, txt_path: str) -> str:
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            return f"Error parsing TXT: {e}"

               
        
    