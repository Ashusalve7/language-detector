# FastAPI Integrated Document Processor
# Install: pip install fastapi uvicorn pytesseract pillow PyPDF2 python-docx transformers torch sentencepiece PyMuPDF googletrans==4.0.0-rc1

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import re
import base64
import requests
from PIL import Image
import pytesseract
import PyPDF2
import docx
import fitz
from io import BytesIO
from transformers import pipeline
from googletrans import Translator
import tempfile
import os

class LanguageIdentifier:
    def __init__(self):
        self.malayalam_pattern = re.compile(r'[\u0d00-\u0d7f]+')
        self.english_pattern = re.compile(r'[a-zA-Z\s.,!?;:"\'()\-]+')

    def extract_text_from_image(self, image_bytes):
        try:
            image = Image.open(BytesIO(image_bytes))
            text = pytesseract.image_to_string(image, lang='eng+mal')
            return text.strip()
        except Exception as e:
            print(f"Error in OCR: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_bytes):
        try:
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            
            # If direct extraction yields little text, use OCR
            if len(text.strip()) < 100:
                print("Low text content detected. Switching to OCR mode for PDF...")
                text = ""
                doc = fitz.open("pdf", pdf_bytes)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    image = Image.open(BytesIO(img_data))
                    text += pytesseract.image_to_string(image, lang='eng+mal') + "\n"
                doc.close()
            
            return text.strip()
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return ""

    def extract_text_from_docx(self, docx_bytes):
        try:
            document = docx.Document(BytesIO(docx_bytes))
            text = "\n".join([p.text for p in document.paragraphs])
            return text.strip()
        except Exception as e:
            print(f"Error extracting DOCX text: {e}")
            return ""

    def extract_text_from_txt(self, txt_bytes):
        try:
            return txt_bytes.decode("utf-8").strip()
        except Exception as e:
            print(f"Error extracting TXT text: {e}")
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

    def extract_text_from_file(self, file_bytes, ext):
        if ext in ["jpg", "jpeg", "png", "bmp", "tiff"]:
            return self.extract_text_from_image(file_bytes)
        elif ext == "pdf":
            return self.extract_text_from_pdf(file_bytes)
        elif ext == "docx":
            return self.extract_text_from_docx(file_bytes)
        elif ext == "txt":
            return self.extract_text_from_txt(file_bytes)
        else:
            return ""

class MalayalamTranslator:
    def __init__(self):
        self.translator = Translator()
    
    def translate_malayalam_to_english(self, text):
        try:
            print("🔄 Translating Malayalam text to English...")
            result = self.translator.translate(text, src='ml', dest='en')
            return result.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text

class DocumentSummarizer:
    def __init__(self):
        print("Loading summarization model...")
        try:
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            print("✅ Summarization model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.summarizer = None

    def extract_invoice_data(self, text):
        invoice_data = {}
        patterns = {
            'invoice_number': re.compile(r'(?:Invoice|Facture|Receipt|№)\s*(?:No\.?|N°)?\s*[:#]?\s*([A-Z0-9\-]+)', re.IGNORECASE),
            'date': re.compile(r'Date\s*[:]?\s*((?:\d{4}[/-]\d{1,2}[/-]\d{1,2})|(?:\d{1,2}[/-]\d{1,2}[/-]\d{4}))', re.IGNORECASE),
            'total_amount': re.compile(r'(?:Total\s*Value|Total\s*\(INR\)|Total\s*Amount|Montant\s*TTC|Total)\s*.*?([\d,]+\.\d{2})', re.IGNORECASE)
        }
        
        for key, pattern in patterns.items():
            match = pattern.search(text)
            if match:
                value = match.group(1).strip()
                if key == 'total_amount':
                    value = value.replace(',', '')
                invoice_data[key] = value
        
        lines = text.split('\n')
        line_items = []
        is_capturing_items = False
        start_keywords = ['description', 'product', 'item']
        end_keywords = ['subtotal', 'total', 'taxes', 'balance due', 'my account', 'payment must reach']
        
        for line in lines:
            line_lower = line.lower()
            if is_capturing_items and any(keyword in line_lower for keyword in end_keywords):
                is_capturing_items = False
            if not is_capturing_items and any(keyword in line_lower for keyword in start_keywords):
                is_capturing_items = True
                continue
            if is_capturing_items and line.strip() and len(line.strip()) > 2:
                line_items.append(line.strip())
        
        if line_items:
            invoice_data['line_items'] = line_items
        
        if not invoice_data:
            return None
        
        # Currency detection
        currency_symbol = None
        text_upper = text.upper()
        if '₹' in text or 'INR' in text_upper:
            currency_symbol = '₹'
        elif '€' in text or 'EUR' in text_upper:
            currency_symbol = '€'
        elif '£' in text or 'GBP' in text_upper:
            currency_symbol = '£'
        elif '$' in text or 'USD' in text_upper:
            currency_symbol = '$'
        
        if currency_symbol and 'total_amount' in invoice_data:
            invoice_data['total_amount'] = f"{currency_symbol}{invoice_data['total_amount']}"
        
        return invoice_data

    def extract_circular_data(self, text):
        circular_data = {}
        patterns = {
            'file_number': re.compile(r'No\.\s*(T-[A-Z0-9/\-]+)', re.IGNORECASE),
            'date_of_issue': re.compile(r'Dated\s*[:]?\s*(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s*,\s*\d{4})', re.IGNORECASE),
            'subject': re.compile(r'Sub(?:ject)?\s*[:]?\s*(.*?)(?=Madam/Sir|\n\n)', re.IGNORECASE | re.DOTALL),
            'new_commencement_date': re.compile(r'extended\s+to\s+([\d\w\s,.]+?)\s+instead\s+of', re.IGNORECASE),
            'new_application_deadline': re.compile(r'revised\s+to\s+([\d\w\s,.]+?)\s+instead\s+of', re.IGNORECASE)
        }
        
        for key, pattern in patterns.items():
            match = pattern.search(text)
            if match:
                value = re.sub(r'\s+', ' ', match.group(1)).strip().replace('.', '')
                circular_data[key] = value
        
        return circular_data if circular_data else None

    def classify_document(self, text):
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in ['invoice', 'facture', 'receipt']):
            return 'invoice'
        if any(keyword in text_lower for keyword in ['government of india', 'ministry of', 'circular', 'department of']):
            return 'government_circular'
        return 'prose'

    def summarize_text(self, text):
        if self.summarizer is None:
            return "Summarization model is not available."
        
        safe_text = text[:5000]
        word_count = len(safe_text.split())
        
        if word_count < 50:
            return "Text is too short to generate a meaningful summary."
        
        try:
            summary_max_len = min(150, int(word_count * 0.5))
            summary_min_len = min(40, int(word_count * 0.2))
            
            if summary_min_len >= summary_max_len:
                summary_min_len = int(summary_max_len * 0.5)
            
            summary = self.summarizer(safe_text, max_length=summary_max_len, min_length=summary_min_len, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            return f"Could not generate summary. Error: {e}"

    def process_text(self, text):
        if not text or not text.strip():
            return {"error": "No text content found in the document."}
        
        safe_text = text[:5000]
        doc_type = self.classify_document(safe_text)
        
        result = {
            "document_type": doc_type,
            "structured_data": None,
            "summary": None
        }
        
        # Extract structured data
        if doc_type == 'invoice':
            result["structured_data"] = self.extract_invoice_data(safe_text)
        elif doc_type == 'government_circular':
            result["structured_data"] = self.extract_circular_data(safe_text)
        
        # Generate summary
        result["summary"] = self.summarize_text(safe_text)
        
        return result

class IntegratedDocumentProcessor:
    def __init__(self):
        print("🚀 Initializing Integrated Document Processor...")
        self.language_identifier = LanguageIdentifier()
        self.translator = MalayalamTranslator()
        self.summarizer = DocumentSummarizer()
        print("✅ All components loaded successfully!")
    
    def process_file_bytes(self, file_bytes, filename):
        """Main processing pipeline for file bytes"""
        file_extension = filename.split('.')[-1].lower()
        
        # Step 1: Extract text from file
        extracted_text = self.language_identifier.extract_text_from_file(file_bytes, file_extension)
        
        if not extracted_text or not extracted_text.strip():
            return {"error": "Could not extract any text from the document."}
        
        # Step 2: Identify language
        language_result = self.language_identifier.identify_language(extracted_text)
        detected_language = language_result.get('language', 'unknown')
        
        # Step 3: Process based on language
        text_to_process = extracted_text
        translation_performed = False
        
        if detected_language == 'malayalam':
            translated_text = self.translator.translate_malayalam_to_english(extracted_text)
            text_to_process = translated_text
            translation_performed = True
        
        # Step 4: Summarize and extract structured data
        processing_result = self.summarizer.process_text(text_to_process)
        
        # Combine all results
        final_result = {
            "file_info": {
                "filename": filename,
                "file_type": file_extension.upper(),
                "language_detected": detected_language,
                "confidence": language_result.get('confidence', 0),
                "translation_performed": translation_performed
            },
            "processing_result": processing_result,
            "raw_text": extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text
        }
        
        return final_result

# Initialize the integrated processor
processor = IntegratedDocumentProcessor()

# FastAPI App
app = FastAPI(
    title="Integrated Document Processor",
    description="Language detection, translation, and intelligent document processing API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Integrated Document Processor API",
        "features": [
            "Language Detection (English/Malayalam)",
            "Automatic Translation (Malayalam → English)",
            "Smart Document Classification",
            "Structured Data Extraction",
            "Intelligent Summarization"
        ],
        "supported_formats": ["txt", "pdf", "docx", "jpg", "png", "jpeg"]
    }

@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    """Process uploaded document through the integrated pipeline"""
    try:
        # Validate file type
        file_extension = file.filename.split('.')[-1].lower()
        supported_types = ["txt", "pdf", "docx", "jpg", "jpeg", "png", "bmp", "tiff"]
        
        if file_extension not in supported_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Supported types: {supported_types}"
            )
        
        # Read file bytes
        file_bytes = await file.read()
        
        # Process through integrated pipeline
        result = processor.process_file_bytes(file_bytes, file.filename)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/process/base64")
async def process_base64_document(filename: str = Form(...), content: str = Form(...)):
    """Process document from base64 encoded content"""
    try:
        # Validate file type
        file_extension = filename.split('.')[-1].lower()
        supported_types = ["txt", "pdf", "docx", "jpg", "jpeg", "png", "bmp", "tiff"]
        
        if file_extension not in supported_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Supported types: {supported_types}"
            )
        
        # Decode base64 content
        file_bytes = base64.b64decode(content)
        
        # Process through integrated pipeline
        result = processor.process_file_bytes(file_bytes, filename)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/process/url")
async def process_document_from_url(url: str = Form(...)):
    """Process document from URL"""
    try:
        # Download file from URL
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download file from URL")
        
        # Extract filename from URL
        filename = url.split("/")[-1].split("?")[0]
        if not filename or '.' not in filename:
            raise HTTPException(status_code=400, detail="Could not determine file type from URL")
        
        file_extension = filename.split('.')[-1].lower()
        supported_types = ["txt", "pdf", "docx", "jpg", "jpeg", "png", "bmp", "tiff"]
        
        if file_extension not in supported_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Supported types: {supported_types}"
            )
        
        # Process through integrated pipeline
        result = processor.process_file_bytes(response.content, filename)
        
        return JSONResponse(content=result)
        
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/identify")
async def identify_language_only(file: UploadFile = File(...)):
    """Only identify language without full processing"""
    try:
        file_extension = file.filename.split('.')[-1].lower()
        file_bytes = await file.read()
        
        # Extract text
        extracted_text = processor.language_identifier.extract_text_from_file(file_bytes, file_extension)
        
        if not extracted_text:
            return {"error": "Could not extract text from file"}
        
        # Identify language
        language_result = processor.language_identifier.identify_language(extracted_text)
        language_result["file_type"] = file_extension
        language_result["text_sample"] = extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
        
        return JSONResponse(content=language_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Language identification error: {str(e)}")

@app.post("/translate")
async def translate_malayalam_text(text: str = Form(...)):
    """Translate Malayalam text to English"""
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Check if text is Malayalam
        language_result = processor.language_identifier.identify_language(text)
        
        if language_result["language"] != "malayalam":
            return {
                "original_text": text,
                "translated_text": text,
                "message": "Text does not appear to be Malayalam",
                "detected_language": language_result["language"]
            }
        
        # Translate
        translated_text = processor.translator.translate_malayalam_to_english(text)
        
        return {
            "original_text": text,
            "translated_text": translated_text,
            "detected_language": "malayalam",
            "confidence": language_result["confidence"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

@app.post("/process/simple")
async def process_document_simple(file: UploadFile = File(...)):
    """Simplified endpoint optimized for n8n - returns flattened structure"""
    try:
        file_extension = file.filename.split('.')[-1].lower()
        supported_types = ["txt", "pdf", "docx", "jpg", "jpeg", "png", "bmp", "tiff"]
        
        if file_extension not in supported_types:
            return {"success": False, "error": f"Unsupported file type: {file_extension}"}
        
        file_bytes = await file.read()
        result = processor.process_file_bytes(file_bytes, file.filename)
        
        if "error" in result:
            return {"success": False, "error": result["error"]}
        
        # Flatten structure for easier n8n parsing
        flattened = {
            "success": True,
            "filename": result["file_info"]["filename"],
            "file_type": result["file_info"]["file_type"],
            "language": result["file_info"]["language_detected"],
            "confidence": result["file_info"]["confidence"],
            "was_translated": result["file_info"]["translation_performed"],
            "document_type": result["processing_result"]["document_type"],
            "summary": result["processing_result"]["summary"],
            "raw_text_preview": result["raw_text"]
        }
        
        # Add structured data fields if available
        if result["processing_result"]["structured_data"]:
            structured = result["processing_result"]["structured_data"]
            for key, value in structured.items():
                if key != "line_items":  # Handle line items separately
                    flattened[f"extracted_{key}"] = value
                else:
                    flattened["extracted_items"] = value
        
        return flattened
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/process/webhook")
async def process_webhook(
    file_url: str = Form(None),
    file_base64: str = Form(None),
    filename: str = Form(...)
):
    """Webhook-friendly endpoint for n8n integrations"""
    try:
        if file_url:
            # Process from URL
            response = requests.get(file_url, timeout=30)
            if response.status_code != 200:
                return {"success": False, "error": "Failed to download file from URL"}
            file_bytes = response.content
            
        elif file_base64:
            # Process from base64
            file_bytes = base64.b64decode(file_base64)
            
        else:
            return {"success": False, "error": "Either file_url or file_base64 must be provided"}
        
        result = processor.process_file_bytes(file_bytes, filename)
        
        if "error" in result:
            return {"success": False, "error": result["error"]}
        
        # Return n8n-friendly format
        return {
            "success": True,
            "data": {
                "file_info": result["file_info"],
                "processing": result["processing_result"],
                "text_preview": result["raw_text"]
            },
            "message": f"Successfully processed {filename}"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/n8n/info")
async def n8n_integration_info():
    """Information for n8n integration"""
    return {
        "service": "Integrated Document Processor",
        "version": "1.0.0",
        "n8n_compatible": True,
        "endpoints": {
            "simple_processing": {
                "url": "/process/simple",
                "method": "POST",
                "description": "Upload file, get flattened response",
                "content_type": "multipart/form-data"
            },
            "webhook_processing": {
                "url": "/process/webhook", 
                "method": "POST",
                "description": "Process via URL or base64",
                "content_type": "application/x-www-form-urlencoded",
                "parameters": {
                    "filename": "required",
                    "file_url": "optional - URL to download file",
                    "file_base64": "optional - base64 encoded file"
                }
            },
            "language_only": {
                "url": "/identify",
                "method": "POST", 
                "description": "Only identify language",
                "content_type": "multipart/form-data"
            }
        },
        "response_format": "JSON with success/error flags",
        "supported_files": ["txt", "pdf", "docx", "jpg", "png", "jpeg"],
        "features": [
            "Auto language detection (English/Malayalam)",
            "Malayalam to English translation", 
            "Smart document classification",
            "Structured data extraction",
            "Text summarization"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "language_identifier": "loaded",
            "translator": "loaded",
            "summarizer": "loaded" if processor.summarizer.summarizer is not None else "failed"
        },
        "n8n_ready": True
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting n8n-Compatible Document Processor API...")
    print("📋 n8n Integration endpoints:")
    print("   • POST /process/simple - Simplified processing")
    print("   • POST /process/webhook - Webhook-friendly") 
    print("   • GET /n8n/info - Integration information")
    uvicorn.run(app, host="0.0.0.0", port=8000)