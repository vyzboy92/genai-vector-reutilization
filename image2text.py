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

def process_images(image_folder, output_file, prompt):
    """
    Processes images in a folder, generating captions for each using Ollama.

    Args:
        image_folder (str): Path to the folder containing images.
        output_file (str): Path to the file to append the generated captions.
        prompt (str): Prompt for generating captions.
    """
    image_count = 0

    with open(output_file, 'a') as output:
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

                # Append the captions to the output file
                for caption in captions:
                    output.write(caption + '\n')

                logging.info(f"Captions appended for {filename}.")

                image_count += 1
                logging.info(f"Processed {image_count} image(s).")

            except Exception as e:
                logging.error(f"Error processing image {file_path}: {e}")

if __name__ == "__main__":
    IMAGE_FOLDER = "./images"
    OUTPUT_FILE = "./output/captions.txt"
    PROMPT = (
        "As a driver driving through Europe, describe the scene and all the objects "
        "you see in the image in one sentence."
    )

    logging.info("Starting image processing script.")
    process_images(IMAGE_FOLDER, OUTPUT_FILE, PROMPT)
    logging.info("Image processing completed.")
