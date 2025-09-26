import re
from PIL import Image
import pytesseract
import PyPDF2
import docx
from io import BytesIO

class LanguageIdentifier:
    def __init__(self):
        self.malayalam_pattern = re.compile(r'[\u0d00-\u0d7f]+')
        self.english_pattern = re.compile(r'[a-zA-Z\s.,!?;:"\'()\-]+')
        
    def extract_text_from_image(self, image_bytes):
        try:
            image = Image.open(BytesIO(image_bytes))
            text = pytesseract.image_to_string(image, lang='eng+mal')
            return text.strip()
        except Exception:
            return ""
    
    def extract_text_from_pdf(self, pdf_bytes):
        try:
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text.strip()
        except:
            return ""
    
    def extract_text_from_docx(self, docx_bytes):
        try:
            document = docx.Document(BytesIO(docx_bytes))
            text = "\n".join([p.text for p in document.paragraphs])
            return text.strip()
        except:
            return ""
    
    def extract_text_from_txt(self, txt_bytes):
        try:
            return txt_bytes.decode("utf-8").strip()
        except:
            return ""
    
    def identify_language(self, text):
        if not text or len(text.strip()) < 3:
            return {"language": "unknown", "confidence": 0.0, "reason": "Text too short or empty"}
        
        malayalam_matches = self.malayalam_pattern.findall(text)
        malayalam_ratio = sum(len(m) for m in malayalam_matches) / len(text)
        english_matches = self.english_pattern.findall(text)
        english_ratio = sum(len(m) for m in english_matches) / len(text)
        
        if malayalam_ratio > 0.05:
            return {"language": "malayalam", "confidence": round(min(malayalam_ratio * 3, 1.0), 2),
                    "reason": f"Malayalam characters found: {malayalam_ratio:.1%}"}
        elif english_ratio > 0.3:
            return {"language": "english", "confidence": round(min(english_ratio, 1.0), 2),
                    "reason": f"English characters found: {english_ratio:.1%}"}
        else:
            return {"language": "unknown", "confidence": 0.0, "reason": "Unable to determine language"}
    
    def process_file(self, file_bytes, ext):
        if ext in ["jpg", "jpeg", "png", "bmp", "tiff"]:
            text = self.extract_text_from_image(file_bytes)
        elif ext == "pdf":
            text = self.extract_text_from_pdf(file_bytes)
        elif ext == "docx":
            text = self.extract_text_from_docx(file_bytes)
        elif ext == "txt":
            text = self.extract_text_from_txt(file_bytes)
        else:
            return {"error": "Unsupported file type", "supported_types": ["jpg", "png", "pdf", "docx", "txt"]}
        
        result = self.identify_language(text)
        result["file_type"] = ext
        return result
