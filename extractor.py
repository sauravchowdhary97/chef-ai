from dotenv import load_dotenv
from langchain.tools import tool
from PIL import Image
import pytesseract
import json
import requests
import os
import re

# Load environment variables
load_dotenv()

# Set up API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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


@tool
def parse_recipe_text(text: str) -> str:
    """Parse raw recipe text into structured JSON format"""
    try:
        # Using OpenAI to parse the recipe
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        payload = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "system", 
                    "content": "Parse the following recipe text into a structured JSON with the following fields: title, description, servings, ingredients (as a list with quantity, unit, and name), and steps (as a numbered list). Make sure to maintain the exact quantities and instructions."
                },
                {
                    "role": "user", 
                    "content": text
                }
            ],
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error parsing recipe: {str(e)}"

@tool
def identify_cooking_actions(recipe_json_str: str) -> str:
    """Identify cooking actions from recipe steps"""
    try:
        recipe_json = json.loads(recipe_json_str)
        cooking_verbs = ["chop", "dice", "slice", "mince", "grate", "mix", "stir", "beat", "whisk", 
                        "fold", "bake", "roast", "grill", "boil", "simmer", "fry", "sauté", "steam",
                        "poach", "marinate", "season", "sprinkle", "pour", "drizzle", "blend", "puree"]
        
        actions = []
        for i, step in enumerate(recipe_json["steps"]):
            step_actions = []
            for verb in cooking_verbs:
                if re.search(r'\b' + verb + r'\b', step.lower()):
                    step_actions.append({
                        "action": verb,
                        "step_number": i + 1,
                        "full_instruction": step
                    })
            actions.extend(step_actions)
        
        return json.dumps(actions)
    except Exception as e:
        return f"Error identifying cooking actions: {str(e)}"