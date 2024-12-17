# LLM Sentence generation

import base64
import os
import re
from io import BytesIO
import logging

import ollama
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/image2text.log"),
        logging.StreamHandler()
    ]
)

def load_image(pil_image_path):
    """
    Loads an image, converts it to base64 format.

    Args:
        pil_image_path (str): Path to the image file.

    Returns:
        str: Base64 encoded string of the image.
    """
    try:
        pil_image = Image.open(pil_image_path)
        image_b64 = convert_to_base64(pil_image)
        logging.info(f"Successfully loaded and converted image: {pil_image_path}")
        return image_b64
    except Exception as e:
        logging.error(f"Error loading image {pil_image_path}: {e}")
        raise

def convert_to_base64(pil_image):
    """
    Converts a PIL image to a base64 encoded string.

    Args:
        pil_image (PIL.Image): PIL image object.

    Returns:
        str: Base64 encoded string of the image.
    """
    try:
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    except Exception as e:
        logging.error(f"Error converting image to base64: {e}")
        raise

def process_images(image_folder, output_folder, prompt):
    """
    Processes images in a folder, generating captions for each using Ollama.

    Args:
        image_folder (str): Path to the folder containing images.
        output_folder (str): Path to the folder to save the output text files.
        prompt (str): Prompt for generating captions.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logging.info(f"Created output folder: {output_folder}")

    image_count = 0

    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)
        if not os.path.isfile(file_path):
            logging.warning(f"Skipping non-file item: {file_path}")
            continue

        try:
            logging.info(f"Processing image: {file_path}")
            image_b64 = load_image(file_path)

            # Generate caption using Ollama
            response = ollama.generate(
                model='llava',
                prompt=prompt,
                images=[image_b64]
            )
            captions = re.split(r'(?<=[.!?])\s+', response['response'].strip())

            # Save the captions to a text file
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
            with open(output_file_path, 'w') as output_file:
                for caption in captions:
                    output_file.write(caption + '\n')

            logging.info(f"Captions saved for {filename} at {output_file_path}")

            image_count += 1
            logging.info(f"Processed {image_count} image(s).")

        except Exception as e:
            logging.error(f"Error processing image {file_path}: {e}")

if __name__ == "__main__":
    IMAGE_FOLDER = "./images"
    OUTPUT_FOLDER = "./output"
    PROMPT = (
        "As a driver driving through Europe, describe the scene and all the objects "
        "you see in the image in one sentence."
    )

    logging.info("Starting image processing script.")
    process_images(IMAGE_FOLDER, OUTPUT_FOLDER, PROMPT)
    logging.info("Image processing completed.")
