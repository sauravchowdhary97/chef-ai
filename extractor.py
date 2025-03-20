from dotenv import load_dotenv
from langchain.tools import tool
from PIL import Image
import pytesseract

# Load environment variables
load_dotenv()

# Tool definitions
@tool
def extract_text_from_image(image_path: str) -> str:
    """Extract text from a recipe image using OCR"""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"