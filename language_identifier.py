import re
import os
from PIL import Image
import pytesseract
import PyPDF2
import docx

class LanguageIdentifier:
    def __init__(self):
        self.malayalam_pattern = re.compile(r'[\u0d00-\u0d7f]+')
        self.english_pattern = re.compile(r'[a-zA-Z\s.,!?;:"\'()\-]+')
        
    def extract_text_from_image(self, image_path):
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang='eng+mal')
            return text.strip()
        except Exception as e:
            return ""
    
    def extract_text_from_pdf(self, pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text.strip()
        except:
            return ""
    
    def extract_text_from_docx(self, docx_path):
        try:
            doc = docx.Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except:
            return ""
    
    def extract_text_from_txt(self, txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except:
            return ""
    
    def identify_language(self, text):
        if not text or len(text.strip()) < 3:
            return {"language": "unknown","confidence": 0.0,"reason": "Text too short or empty"}
        
        text = text.strip()
        malayalam_matches = self.malayalam_pattern.findall(text)
        malayalam_chars = sum(len(match) for match in malayalam_matches)
        malayalam_ratio = malayalam_chars / len(text) if len(text) > 0 else 0
        english_matches = self.english_pattern.findall(text)
        english_chars = sum(len(match) for match in english_matches)
        english_ratio = english_chars / len(text) if len(text) > 0 else 0
        
        if malayalam_ratio > 0.05:
            confidence = min(malayalam_ratio * 3, 1.0)
            return {"language": "malayalam","confidence": round(confidence, 2),"reason": f"Malayalam characters found: {malayalam_ratio:.1%}"}
        elif english_ratio > 0.3:
            confidence = min(english_ratio, 1.0)
            return {"language": "english","confidence": round(confidence, 2),"reason": f"English characters found: {english_ratio:.1%}"}
        else:
            return {"language": "unknown","confidence": 0.0,"reason": "Unable to determine language"}
    
    def process_file(self, file_path):
        try:
            ext = file_path.split('.')[-1].lower()
            if ext in ['jpg','jpeg','png','bmp','tiff']:
                text = self.extract_text_from_image(file_path)
            elif ext == 'pdf':
                text = self.extract_text_from_pdf(file_path)
            elif ext == 'docx':
                text = self.extract_text_from_docx(file_path)
            elif ext == 'txt':
                text = self.extract_text_from_txt(file_path)
            else:
                return {"error": "Unsupported file type","supported_types": ["jpg, png, pdf, docx, txt"]}
            
            result = self.identify_language(text)
            result["file_type"] = ext
            return result
        except Exception as e:
            return {"error": str(e),"language": "unknown","confidence": 0.0}
