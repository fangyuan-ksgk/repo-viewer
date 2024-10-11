import ast
import io
import sys
import base64
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import subprocess
from PIL import Image as PILImage
from IPython.display import display, Image
from typing import Optional
from .repo import parse_python_code, parse_response


def base64_to_pil_image(base64_str):
    # Decode the base64 string to bytes
    img_data = base64.b64decode(base64_str)
    
    # Create a BytesIO object from the bytes
    img_buffer = io.BytesIO(img_data)
    
    # Open the image using PIL
    pil_image = Image.open(img_buffer)
    
    return pil_image

def combine_figures(figures) -> Optional[PILImage.Image]:

    # Check if there are figures to combine
    if not figures:
        return None

    # Convert base64 encoded figures to PIL Images
    pil_images = []
    for fig in figures:
        img_data = base64.b64decode(fig)
        img = PILImage.open(io.BytesIO(img_data))
        pil_images.append(img)

    # Calculate the total width and maximum height
    total_width = sum(img.width for img in pil_images)
    max_height = max(img.height for img in pil_images)

    # Create a new image with the calculated dimensions
    new_image = PILImage.new('RGB', (total_width, max_height))

    # Paste all images into the new image
    current_width = 0
    for img in pil_images:
        new_image.paste(img, (current_width, 0))
        current_width += img.width

    # Save the combined image to a BytesIO object
    combined_buffer = io.BytesIO()
    new_image.save(combined_buffer, format='PNG')
    combined_buffer.seek(0)

    # Encode the combined image to base64
    combined_figure = base64.b64encode(combined_buffer.getvalue()).decode()

    # Return the combined figure as a base64 encoded string
    return base64_to_pil_image(combined_figure)
    

class CodeInterpreter:
    def __init__(self):
        self.locals = {}
        self.globals = globals().copy()
        self.output = []
        self.figures = []
        
    def __call__(self, response: str):
        """ 
        Two situation: 
        - We have code-snippet, compile it, flag (True)
        - Otherwise, this is a direct response, return it with a flag (False)
        """
        code, is_code = self.parse_code(response)
        if is_code:
            self.execute(code)
            return self.return_results(), self.figure, True

        return parse_response(response), self.figure, False
        
        
    def parse_code(self, response: str):
        code = parse_python_code(response)
        if code:
            return code, True
        return response, False

    def execute(self, code):        
        # Reset output and figures
        self.output = []
        self.figures = []

        try:
            # Use subprocess to run the code
            result = subprocess.run(['python', '-c', code], capture_output=True, text=True, check=True)
            self.output.append(result.stdout)

            # TODO: Handle figure generation
            # This part needs to be adjusted as subprocess doesn't directly capture matplotlib figures

        except subprocess.CalledProcessError as e:
            pass
        except Exception as e:
            pass
        
    @property 
    def figure(self):
        return combine_figures(self.figures)

    def get_output(self):
        return "\n".join(self.output)

    def get_figures(self):
        return self.figures

    def display_results(self):
        print(self.get_output())
        for fig in self.get_figures():
            display(Image(data=base64.b64decode(fig)))
            
    def return_results(self):
        compiled_result = self.get_output()
        return compiled_result