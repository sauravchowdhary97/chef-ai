from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain.tools import tool
from PIL import Image
import pytesseract
import json
import requests
import tempfile
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

@tool
def extract_durations(recipe_json_str: str) -> str:
    """Extract cooking durations from recipe steps"""
    try:
        recipe_json = json.loads(recipe_json_str)
        time_pattern = r'(\d+)(?:\s*-\s*\d+)?\s*(minute|minutes|mins|min|hour|hours|hr|hrs)'
        durations = []
        
        for i, step in enumerate(recipe_json["steps"]):
            matches = re.finditer(time_pattern, step, re.IGNORECASE)
            for match in matches:
                time_value = int(match.group(1))
                unit = match.group(2).lower()
                
                # Convert to minutes for consistency
                if unit.startswith("hour") or unit == "hr" or unit == "hrs":
                    time_value *= 60
                
                durations.append({
                    "step_number": i + 1,
                    "duration_minutes": time_value,
                    "original_text": match.group(0),
                    "full_instruction": step
                })
        
        return json.dumps(durations)
    except Exception as e:
        return f"Error extracting durations: {str(e)}"

@tool
def generate_video_prompt(step: str, step_number: int, ingredients_json: str) -> str:
    """Generate a detailed prompt for AI video generation"""
    try:
        ingredients = json.loads(ingredients_json)
        
        # Extract the main cooking action
        cooking_verbs = ["chop", "dice", "slice", "mince", "grate", "mix", "stir", "beat", "whisk", 
                        "fold", "bake", "roast", "grill", "boil", "simmer", "fry", "sauté", "steam"]
        main_action = "cooking"
        for action in cooking_verbs:
            if action in step.lower():
                main_action = action
                break
        
        # Extract relevant ingredients for this step
        relevant_ingredients = []
        for ingredient in ingredients:
            ingredient_name = ingredient.get("name", "").lower()
            if ingredient_name in step.lower():
                relevant_ingredients.append(ingredient_name)
        
        ingredients_list = ", ".join(relevant_ingredients) if relevant_ingredients else "all necessary ingredients"
        
        # Generate the prompt
        prompt_template = f"""
        Create a realistic, top-down view of {main_action}. 
        Kitchen setting with clean countertop, good lighting. 
        
        Specifically show: {step}
        
        Ingredients visible: {ingredients_list}
        
        Style: Professional cooking video with a homely feeling, 4K quality, soft natural lighting.
        """
        
        return prompt_template
    except Exception as e:
        return f"Error generating video prompt: {str(e)}"


# Define the agents
ocr_agent = Agent(
    role="OCR Specialist",
    goal="Extract text accurately from recipe images",
    backstory=f"""
        You are an expert in optical character recognition with years of experience in extracting text from various types of images. 
        Your specialty is recipe images, where you can identify ingredients, instructions, and other recipe components.
        """,
    verbose=True,
    allow_delegation=True,
    tools=[extract_text_from_image]
)


recipe_parser_agent = Agent(
    role="Recipe Parser",
    goal="Transform raw recipe text into structured data",
    backstory=f"""
        You are a culinary data specialist who excels at understanding recipe formats and converting them into structured, machine-readable formats. 
        You understand cooking terminology and recipe organization.
        """,
    verbose=True,
    allow_delegation=True,
    tools=[parse_recipe_text, identify_cooking_actions, extract_durations]
)

# Define the tasks
extract_text_task = Task(
    description="Extract text from the recipe image",
    agent=ocr_agent,
    expected_output="The raw text extracted from the recipe image"
)

parse_recipe_task = Task(
    description="Parse the extracted text into structured recipe data",
    agent=recipe_parser_agent,
    expected_output="A JSON object containing recipe title, ingredients, steps, and other structured data"
)

analyze_recipe_task = Task(
    description="Identify cooking actions and extract cooking durations from the recipe.",
    agent=recipe_parser_agent,
    expected_output="A JSON object containing cooking actions and durations for each step"
)

# Create the crew
recipe_video_crew = Crew(
    agents=[ocr_agent, recipe_parser_agent],
    tasks=[extract_text_task, parse_recipe_task, analyze_recipe_task],
    verbose=2,
    process=Process.sequential  # Tasks will be executed in order
)

# Main application function
def convert_img_to_text_recipe(image_path, output_dir="./output"):
    """Convert a recipe image to a cooking video using CrewAI (future iteration)"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up temp directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set context for the crew
        context = {
            "image_path": image_path,
            "temp_dir": temp_dir,
            "output_dir": output_dir
        }
        
        # Execute the crew's tasks
        result = recipe_video_crew.kickoff(inputs=context)
        
        return result

if __name__ == "__main__":
    # Example usage
    recipe_image = "./sample_recipe.jpg"
    result = convert_img_to_text_recipe(recipe_image)
    print(f"Final result: {result}")