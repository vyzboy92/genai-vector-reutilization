# LLM Sentence generation

import base64
import os
import re
from io import BytesIO

import ollama
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer


def load_image(pil_image_path):
    pil_image = Image.open(pil_image_path)
    image_b64 = convert_to_base64(pil_image)
    return image_b64

def convert_to_base64(pil_image: Image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

image_folder = "/content/drive/MyDrive/Research/Images"

prompt = "As a driver driving through Europe, describe the scene and all the objects you see in the image in one sentence"

sentence_list = []
image_count = 1

for filename in os.listdir(image_folder):
  print("\r", end=f"Processing Image Number: {image_count}")
  # Construct the full file path
  file_path = os.path.join(image_folder, filename)
  # Convert the image to a PIL Image
  image_b64 = load_image(file_path)

  # Ollama generate
  resp = ollama.generate(
      model='llava',
      prompt=prompt,
      images=[image_b64]
      )['response']
  sentences = re.split(r'(?<=[.!?])\\s+', resp.strip())
  for sentence in sentences:
    sentence_list.append(sentence)
  image_count += 1

print(sentence_list)

# Define the embedding model (e.g., "all-MiniLM-L6-v2")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode the sentences into embeddings
embeddings = embedding_model.encode(sentence_list, convert_to_tensor=True)

# Save the embeddings as a torch tensor
torch.save(embeddings, "./saved_weights/sentence_embeddings.pt")

# Load the saved tensor (for testing or reuse)
loaded_embeddings = torch.load("sentence_embeddings.pt")

# Verify the shapes
print("Shape of saved embeddings:", embeddings.shape)
print("Shape of loaded embeddings:", loaded_embeddings.shape)