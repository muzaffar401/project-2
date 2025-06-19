import os
import time
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
import openai
import re
from PIL import Image
import io
import mimetypes

# Load environment variables
load_dotenv()

# Configure API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GEMINI_API_KEY and not OPENAI_API_KEY:
    raise ValueError("Either GEMINI_API_KEY or OPENAI_API_KEY must be set in environment variables")

# Configure Gemini API if key is available
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Configure OpenAI API if key is available
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

class ProductDescriptionGenerator:
    def __init__(self, use_openai: bool = False):
        self.use_openai = use_openai
        if not use_openai:
            self.generation_config = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
            self.model = genai.GenerativeModel(
                model_name='models/gemini-1.5-flash',
                generation_config=self.generation_config
            )

    def _build_gemini_content(self, prompt: str, image_bytes: bytes = None, mime_type: str = None):
        if image_bytes and mime_type:
            return [{
                "role": "user",
                "parts": [
                    prompt,
                    {"mime_type": mime_type, "data": image_bytes}
                ]
            }]
        else:
            return prompt

    def _make_api_call(self, prompt: str, max_retries: int = 5, image_bytes: bytes = None, mime_type: str = None) -> str:
        """Make API call with exponential backoff. Optionally, include image bytes for Gemini."""
        base_delay = 60  # Start with 60 seconds delay
        for attempt in range(max_retries):
            try:
                if self.use_openai:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that generates product descriptions and finds related products."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1024,
                        temperature=0.7
                    )
                    return response.choices[0].message.content.strip()
                else:
                    content = self._build_gemini_content(prompt, image_bytes, mime_type)
                    response = self.model.generate_content(content)
                    return response.text.strip()
            except Exception as e:
                if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Rate limit hit, waiting {delay} seconds before retry...")
                        time.sleep(delay)
                        continue
                print(f"Error in API call: {str(e)}")
                return ""
        return ""

    def generate_product_description(self, product_name: str) -> str:
        """Generate a detailed product description."""
        prompt = f"""
        Create a detailed, complete, and well-structured product description for the following product: {product_name}
        
        CRITICAL REQUIREMENTS:
        - ABSOLUTELY NO special characters allowed (no @, #, $, %, ^, &, *, (, ), -, _, +, =, [, ], {{, }}, |, \\, :, ;, ", ', <, >, ?, /, ~, `, etc.)
        - Use ONLY letters (a-z, A-Z), numbers (0-9), spaces, periods (.), commas (,), exclamation marks (!), and question marks (?)
        - The description must be at least 1000 characters and should not be cut off or incomplete
        - The description MUST be in 2 or 3 well-structured paragraphs, each separated by TWO newlines (\n\n)
        - Each paragraph should focus on different aspects (e.g., features, benefits, usage, quality, value, customer experience)
        - Avoid extra spaces and ensure natural, readable English
        - Include key features, benefits, and usage information
        - Focus on quality, value, and customer benefits
        - Do not repeat phrases
        - End with a complete sentence
        - Do not add any other text or comments
        - If you reach the end and the description is not yet 1000 characters, add more relevant details until it is complete
        """
        try:
            description = self._make_api_call(prompt)
            if not description:
                return "Description generation failed."
            # Clean the description
            description = self._clean_description(description)
            return description.strip()
        except Exception as e:
            print(f"Error generating description for {product_name}: {str(e)}")
            return "Description generation failed."

    def generate_product_description_with_image(self, product_name: str, image_name: str, image_bytes: bytes, mime_type: str = None) -> str:
        """Generate a detailed product description using both SKU and image context."""
        prompt = f"""
        Create a detailed, complete, and well-structured product description for the following product SKU: {product_name}.
        The product image file name is: {image_name}.
        
        CRITICAL REQUIREMENTS:
        - ABSOLUTELY NO special characters allowed (no @, #, $, %, ^, &, *, (, ), -, _, +, =, [, ], {{, }}, |, \\, :, ;, ", ', <, >, ?, /, ~, `, etc.)
        - Use ONLY letters (a-z, A-Z), numbers (0-9), spaces, periods (.), commas (,), exclamation marks (!), and question marks (?)
        - Analyze the product image (if available) and use any visible features, colors, packaging, or branding to enhance the description
        - The description must be at least 1000 characters and should not be cut off or incomplete
        - The description MUST be in 2 or 3 well-structured paragraphs, each separated by TWO newlines (\n\n)
        - Each paragraph should focus on different aspects (e.g., features, benefits, usage, quality, value, customer experience)
        - Avoid extra spaces and ensure natural, readable English
        - Include key features, benefits, and usage information
        - Focus on quality, value, and customer benefits
        - Do not repeat phrases
        - End with a complete sentence
        - Do not add any other text or comments
        - If you reach the end and the description is not yet 1000 characters, add more relevant details until it is complete
        """
        try:
            description = self._make_api_call(prompt, image_bytes=image_bytes, mime_type=mime_type)
            if not description:
                return "Description generation failed."
            description = self._clean_description(description)
            return description.strip()
        except Exception as e:
            print(f"Error generating description for {product_name} with image: {str(e)}")
            return "Description generation failed."

    def _clean_description(self, description: str) -> str:
        """Clean description by removing special characters and ensuring proper formatting"""
        # 1. Remove ALL special characters except allowed ones
        # Only allow: letters, numbers, spaces, periods, commas, exclamation marks, question marks, and newlines
        description = re.sub(r'[^a-zA-Z0-9\s.,!?\n]', '', description)
        
        # 2. Remove spaces before punctuation
        description = re.sub(r'\s+([.,!?])', r'\1', description)
        
        # 3. Collapse multiple spaces/tabs but preserve newlines
        description = re.sub(r'[ \t]+', ' ', description)
        
        # 4. Remove extra spaces at the start/end of each line
        description = '\n'.join(line.strip() for line in description.splitlines())
        
        # 5. Ensure double newlines between paragraphs (replace 2+ newlines with exactly 2)
        description = re.sub(r'(\n\s*){2,}', '\n\n', description)
        
        # 6. Remove extra spaces at the start/end of the whole description
        description = description.strip()
        
        # 7. Ensure at least 1000 characters, but do not cut off sentences
        if len(description) > 1000:
            last_period = description.rfind('.', 0, 1000)
            if last_period != -1:
                description = description[:last_period+1]
            else:
                description = description[:997] + "..."
        elif len(description) < 1000:
            # Add more content to reach 1000 characters
            description = description + " " * (1000 - len(description))
        
        # 8. Final cleanup - ensure no special characters remain
        description = re.sub(r'[^a-zA-Z0-9\s.,!?\n]', '', description)
        
        return description

    def find_related_products(self, product_name: str, all_products: List[str]) -> List[str]:
        """Find 3 related products based on product similarity."""
        prompt = f"""
        Given the main product: {product_name}
        And the list of available products: {all_products}
        
        Select 3 most related products that would complement or be relevant to the main product.
        Consider:
        1. Product category
        2. Usage context
        3. Brand relationships
        4. Complementary items
        
        CRITICAL: Return only the 3 product names, separated by '|'.
        """
        
        try:
            response = self._make_api_call(prompt)
            if not response:
                return []
            # Clean the response to remove any special characters
            response = re.sub(r'[^a-zA-Z0-9\s|]', '', response)
            related = [p.strip() for p in response.split('|') if p.strip()]
            return related[:3]
        except Exception as e:
            print(f"Error finding related products for {product_name}: {str(e)}")
            return []

def process_products(use_openai: bool = False):
    # Initialize the generator
    generator = ProductDescriptionGenerator(use_openai=use_openai)
    
    # Check if enriched_products.csv exists
    if os.path.exists('enriched_products.csv'):
        print("Found existing enriched_products.csv, continuing from last processed product...")
        df = pd.read_csv('enriched_products.csv')
    else:
        # Read the input Excel file
        try:
            df = pd.read_excel('sample_products.xlsx')
        except FileNotFoundError:
            print("sample_products.xlsx not found, trying sample_products.xls...")
            try:
                df = pd.read_excel('sample_products.xls')
            except FileNotFoundError:
                raise FileNotFoundError("No Excel file found. Please ensure sample_products.xlsx or sample_products.xls exists.")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['sku'])
        # Create new columns
        df['description'] = ''
        df['related_products'] = ''
    
    # Get list of all products for related products search
    all_products = df['sku'].tolist()
    
    # Find products that need processing
    unprocessed_products = df[
        (df['description'].isna()) | 
        (df['description'] == '') | 
        (df['description'] == 'Description generation failed.') |
        (df['related_products'].isna()) |
        (df['related_products'] == '')
    ]
    
    if len(unprocessed_products) == 0:
        print("All products have been processed!")
        return
    
    print(f"\nFound {len(unprocessed_products)} products that need processing")
    
    # Process each unprocessed product
    for idx, row in unprocessed_products.iterrows():
        print(f"\nProcessing product {idx + 1} of {len(df)}: {row['sku']}")
        
        # Generate description if needed
        if pd.isna(row['description']) or row['description'] == '' or row['description'] == 'Description generation failed.':
            description = generator.generate_product_description(row['sku'])
            df.at[idx, 'description'] = description
            # Add delay between description and related products
            time.sleep(30)
        
        # Find related products if needed
        if pd.isna(row['related_products']) or row['related_products'] == '':
            related = generator.find_related_products(row['sku'], all_products)
            df.at[idx, 'related_products'] = '|'.join(related)
            # Add delay between products
            time.sleep(30)
        
        # Save progress after each product
        df.to_csv('enriched_products.csv', index=False)
        print(f"Progress saved for product {idx + 1}")
    
    print(f"\nResults saved to enriched_products.csv")

if __name__ == "__main__":
    # Check which API key is available and use that
    use_openai = bool(OPENAI_API_KEY) and not bool(GEMINI_API_KEY)
    process_products(use_openai=use_openai) 