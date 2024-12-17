import logging
import torch
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/text2embeddings.log"),
        logging.StreamHandler()
    ]
)

def read_sentences_from_file(filename):
    """
    Reads sentences from a text file and loads them into a list.

    Args:
        filename (str): The name of the text file to read from.

    Returns:
        list: A list of sentences read from the file.
    """
    sentences = []
    try:
        with open(filename, 'r') as file:
            sentences = [line.strip() for line in file if line.strip()]
        logging.info(f"Successfully read {len(sentences)} sentences from {filename}.")
    except FileNotFoundError:
        logging.error(f"The file {filename} does not exist.")
    return sentences

if __name__ == "__main__":

    # Specify the filename
    input_file = "./output/output_sentences.txt"
    
    logging.info("Starting the sentence embedding process.")

    # Define the embedding model (e.g., "all-MiniLM-L6-v2")
    model_name = "all-MiniLM-L6-v2"
    logging.info(f"Loading the embedding model: {model_name}.")
    embedding_model = SentenceTransformer(model_name)

    # Load the sentences from file
    sentences_list = read_sentences_from_file(input_file)
    
    if not sentences_list:
        logging.warning("No sentences found in the file. Exiting process.")
    else:
        # Encode the sentences into embeddings
        logging.info("Encoding sentences into embeddings.")
        embeddings = embedding_model.encode(sentences_list, convert_to_tensor=True)

        # Save the embeddings as a torch tensor
        save_path = "./saved_weights/sentence_embeddings.pt"
        logging.info(f"Saving embeddings to {save_path}.")
        torch.save(embeddings, save_path)

        # Load the saved tensor (for testing or reuse)
        load_path = "sentence_embeddings.pt"
        logging.info(f"Loading embeddings from {load_path}.")
        loaded_embeddings = torch.load(load_path)

        # Verify the shapes
        logging.info(f"Shape of saved embeddings: {embeddings.shape}")
        logging.info(f"Shape of loaded embeddings: {loaded_embeddings.shape}")

    logging.info("Sentence embedding process completed.")
