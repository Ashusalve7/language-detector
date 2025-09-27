# Install required packages
!pip install pytesseract pillow PyPDF2 python-docx transformers torch sentencepiece PyMuPDF googletrans==4.0.0-rc1

# Import libraries
import re
import os
from PIL import Image
import pytesseract
import PyPDF2
import docx
import fitz
import io
from transformers import pipeline
from googletrans import Translator

class LanguageIdentifier:
    def __init__(self):
        # Malayalam Unicode ranges
        self.malayalam_pattern = re.compile(r'[\u0d00-\u0d7f]+')
        # English pattern
        self.english_pattern = re.compile(r'[a-zA-Z\s.,!?;:"\'()\-]+')

    def extract_text_from_image(self, image_path):
        """Extract text from image using OCR"""
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang='eng+mal')
            return text.strip()
        except Exception as e:
            print(f"Error in OCR: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF with fallback to OCR for scanned PDFs"""
        try:
            # First try direct text extraction
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            
            # If direct extraction yields little text, use OCR
            if len(text.strip()) < 100:
                print("Low text content detected. Switching to OCR mode for PDF...")
                text = ""
                with fitz.open(pdf_path) as doc:
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap()
                        img_data = pix.tobytes("png")
                        image = Image.open(io.BytesIO(img_data))
                        text += pytesseract.image_to_string(image, lang='eng+mal') + "\n"
            
            return text.strip()
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return ""

    def extract_text_from_docx(self, docx_path):
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error extracting DOCX text: {e}")
            return ""

    def extract_text_from_txt(self, txt_path):
        """Extract text from TXT file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error extracting TXT text: {e}")
            return ""

    def identify_language(self, text):
        """Identify if text is English or Malayalam"""
        if not text or len(text.strip()) < 3:
            return {
                "language": "unknown",
                "confidence": 0.0,
                "reason": "Text too short or empty"
            }

        text = text.strip()

        # Count Malayalam characters
        malayalam_matches = self.malayalam_pattern.findall(text)
        malayalam_chars = sum(len(match) for match in malayalam_matches)
        malayalam_ratio = malayalam_chars / len(text) if len(text) > 0 else 0

        # Count English characters
        english_matches = self.english_pattern.findall(text)
        english_chars = sum(len(match) for match in english_matches)
        english_ratio = english_chars / len(text) if len(text) > 0 else 0

        # Decision logic
        if malayalam_ratio > 0.05:  # If more than 5% Malayalam characters
            confidence = min(malayalam_ratio * 3, 1.0)
            return {
                "language": "malayalam",
                "confidence": round(confidence, 2),
                "reason": f"Malayalam characters found: {malayalam_ratio:.1%}"
            }
        elif english_ratio > 0.3:  # If more than 30% English characters
            confidence = min(english_ratio, 1.0)
            return {
                "language": "english",
                "confidence": round(confidence, 2),
                "reason": f"English characters found: {english_ratio:.1%}"
            }
        else:
            return {
                "language": "unknown",
                "confidence": 0.0,
                "reason": "Unable to determine language"
            }

    def extract_text_from_file(self, file_path):
        """Extract text from any supported file type"""
        file_extension = file_path.split('.')[-1].lower()

        if file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
            return self.extract_text_from_image(file_path)
        elif file_extension == 'pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension == 'docx':
            return self.extract_text_from_docx(file_path)
        elif file_extension == 'txt':
            return self.extract_text_from_txt(file_path)
        else:
            return ""

class Translator:
    def __init__(self):
        self.translator = Translator()
    
    def translate_malayalam_to_english(self, text):
        """Translate Malayalam text to English"""
        try:
            print("🔄 Translating Malayalam text to English...")
            result = self.translator.translate(text, src='ml', dest='en')
            return result.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text if translation fails

class DocumentSummarizer:
    def __init__(self):
        print("Loading summarization model... (this may take a moment)")
        try:
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            print("✅ Summarization model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.summarizer = None

    def extract_invoice_data(self, text):
        """Extract structured data from invoices"""
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
        
        # Extract line items
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
            invoice_data['line_items'] = "\n".join(line_items)
        
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
        
        output = "📄 **Extracted Invoice Data**:\n"
        for key, value in invoice_data.items():
            if key != 'line_items':
                output += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        if 'line_items' in invoice_data:
            output += f"\n**Items/Description**:\n{invoice_data['line_items']}"
        
        return output

    def extract_circular_data(self, text):
        """Extract structured data from government circulars"""
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
        
        if not circular_data:
            return None
        
        output = "📄 **Extracted Circular Data**:\n"
        for key, value in circular_data.items():
            output += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        return output

    def classify_document(self, text):
        """Classify document type"""
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in ['invoice', 'facture', 'receipt']):
            return 'invoice'
        if any(keyword in text_lower for keyword in ['government of india', 'ministry of', 'circular', 'department of']):
            return 'government_circular'
        return 'prose'

    def summarize_text(self, text):
        """Generate summary of the text"""
        if self.summarizer is None:
            return "⚠️ **Summary Failed**: Summarization model is not available."
        
        safe_text = text[:5000]  # Limit text length for processing
        word_count = len(safe_text.split())
        
        if word_count < 50:
            return "📝 **Summary**: Text is too short to generate a meaningful summary."
        
        try:
            summary_max_len = min(150, int(word_count * 0.5))
            summary_min_len = min(40, int(word_count * 0.2))
            
            if summary_min_len >= summary_max_len:
                summary_min_len = int(summary_max_len * 0.5)
            
            summary = self.summarizer(safe_text, max_length=summary_max_len, min_length=summary_min_len, do_sample=False)
            return f"📝 **Summary**:\n{summary[0]['summary_text']}"
        except Exception as e:
            return f"⚠️ **Summary Failed**: Could not generate a summary. Error: {e}"

    def process_text(self, text):
        """Process text and return structured output"""
        if not text or not text.strip():
            return "❌ Error: No text content found in the document."
        
        safe_text = text[:5000]
        doc_type = self.classify_document(safe_text)
        
        print(f"📋 Document classified as: {doc_type}")
        
        final_output = ""
        
        # Try to extract structured data first
        if doc_type == 'invoice':
            extracted_data = self.extract_invoice_data(safe_text)
            if extracted_data:
                return extracted_data
            else:
                print("Invoice data extraction failed. Falling back to generic summary...")
        elif doc_type == 'government_circular':
            extracted_data = self.extract_circular_data(safe_text)
            if extracted_data:
                final_output += extracted_data + "\n\n"
        
        # Generate summary
        summary = self.summarize_text(safe_text)
        final_output += summary
        
        return final_output.strip()

class IntegratedDocumentProcessor:
    def __init__(self):
        print("🚀 Initializing Integrated Document Processor...")
        self.language_identifier = LanguageIdentifier()
        self.translator = Translator()
        self.summarizer = DocumentSummarizer()
        print("✅ All components loaded successfully!")
    
    def process_file(self, file_path):
        """Main processing pipeline"""
        if not os.path.exists(file_path):
            return "❌ Error: File not found."
        
        print(f"\n📁 Processing file: {file_path}")
        file_extension = file_path.split('.')[-1].lower()
        
        # Step 1: Extract text from file
        print("📖 Step 1: Extracting text from file...")
        extracted_text = self.language_identifier.extract_text_from_file(file_path)
        
        if not extracted_text or not extracted_text.strip():
            return "❌ Error: Could not extract any text from the document."
        
        # Step 2: Identify language
        print("🔍 Step 2: Identifying language...")
        language_result = self.language_identifier.identify_language(extracted_text)
        detected_language = language_result.get('language', 'unknown')
        confidence = language_result.get('confidence', 0)
        
        print(f"🗣️ Language detected: {detected_language.upper()} (Confidence: {confidence:.0%})")
        
        # Step 3: Process based on language
        if detected_language == 'malayalam':
            print("🔄 Step 3: Malayalam detected - translating to English...")
            translated_text = self.translator.translate_malayalam_to_english(extracted_text)
            text_to_process = translated_text
            print("✅ Translation completed!")
        elif detected_language == 'english':
            print("✅ Step 3: English detected - proceeding directly to summarization...")
            text_to_process = extracted_text
        else:
            print("⚠️ Step 3: Language unknown - proceeding with original text...")
            text_to_process = extracted_text
        
        # Step 4: Summarize and extract structured data
        print("📝 Step 4: Processing and summarizing text...")
        final_result = self.summarizer.process_text(text_to_process)
        
        # Prepare final output with metadata
        output = f"""
🎯 **PROCESSING RESULTS**
========================

📋 **File Information:**
- File Type: {file_extension.upper()}
- Language: {detected_language.upper()} ({confidence:.0%} confidence)
{'- Translation: Malayalam → English' if detected_language == 'malayalam' else ''}

{final_result}
        """.strip()
        
        return output

# Initialize the integrated processor
processor = IntegratedDocumentProcessor()

def run_integrated_app():
    """Main application function"""
    print("\n" + "="*50)
    print("🚀 INTEGRATED DOCUMENT PROCESSOR")
    print("="*50)
    print("Features:")
    print("• Language Detection (English/Malayalam)")
    print("• Automatic Translation (Malayalam → English)")
    print("• Smart Document Classification")
    print("• Structured Data Extraction")
    print("• Intelligent Summarization")
    print("\nSupported files: .txt, .pdf, .docx, .png, .jpg, .jpeg")
    print("="*50)
    
    # Upload file
    print("\n📤 Please upload your document:")
    uploaded_files = files.upload()
    
    if not uploaded_files:
        print("❌ No file uploaded. Operation cancelled.")
        return
    
    # Process each uploaded file
    for filename, content in uploaded_files.items():
        print(f"\n🎯 Processing: {filename}")
        
        # Save the uploaded file temporarily
        with open(filename, 'wb') as f:
            f.write(content)
        
        try:
            # Process the file through the integrated pipeline
            result = processor.process_file(filename)
            
            # Display results
            print("\n" + "="*60)
            print("✅ FINAL RESULTS")
            print("="*60)
            print(result)
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error processing file: {e}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(filename):
                os.remove(filename)

# Run the integrated application
if __name__ == "__main__":
    run_integrated_app()

# For programmatic use
def process_document(file_path):
    """Function for direct use in other scripts"""
    return processor.process_file(file_path)