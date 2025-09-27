from transformers import pipeline
import sys

print("⏳ Starting model download...")
try:
    # This line downloads and caches the model weights
    pipeline("summarization", model="facebook/bart-large-cnn")
    print("✅ Summarization model downloaded and cached successfully.")
except Exception as e:
    print(f"❌ Error downloading model during setup: {e}")
    # Exit with a non-zero status to fail the deployment if model setup fails
    sys.exit(1)
