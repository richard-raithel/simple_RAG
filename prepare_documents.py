import os
import fitz  # PyMuPDF
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import re
import numpy as np
import logging
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MongoDB client and database
client = MongoClient(os.environ.get("MONGO_URI"))
db = client[os.environ.get("MONGO_DB")]
collection = db["product_documents"]

# Load the sentence transformer model
model = SentenceTransformer('all-MPNet-base-v2')  # all-MiniLM-L6-v2

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stop words list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def extract_text_from_pdf(pdf_path):
    try:
        document = fitz.open(pdf_path)
        text = ""
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return None


def preprocess_text(text):
    try:
        # Lowercase text
        text = text.lower()

        # Remove unwanted characters, but keep useful punctuation
        text = re.sub(r'[^\w\s.,:;()&-]', '', text)

        # Tokenize text
        words = word_tokenize(text)

        # Remove stop words and lemmatize words
        processed_words = [
            lemmatizer.lemmatize(word) for word in words if word not in stop_words
        ]

        # Rejoin words into a single string
        return ' '.join(processed_words)
    except Exception as e:
        logging.error(f"Error preprocessing text: {e}")
        return None


def encode_text(text):
    try:
        return model.encode([text])[0]
    except Exception as e:
        logging.error(f"Error encoding text: {e}")
        return None


def document_exists(document_id):
    try:
        return collection.count_documents({"document_id": document_id}, limit=1) != 0
    except Exception as e:
        logging.error(f"Error checking if document {document_id} exists: {e}")
        return False


def store_document_in_mongodb(document_id, text, text_embedding, id_embedding):
    if document_exists(document_id):
        logging.info(f"Document {document_id} already exists in MongoDB. Skipping upload.")
        return
    try:
        document = {
            "document_id": document_id,
            "text": text,
            "text_embedding": text_embedding.tolist(),  # Convert numpy array to list for storage
            "id_embedding": id_embedding.tolist()  # Convert numpy array to list for storage
        }
        collection.insert_one(document)
        logging.info(f"Stored document {document_id} in MongoDB.")
    except Exception as e:
        logging.error(f"Error storing document {document_id} in MongoDB: {e}")


def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def remove_text_after_phrase(text, phrase):
    pattern = re.escape(phrase) + ".*"
    return re.sub(pattern, '', text, flags=re.DOTALL).strip()


def process_documents(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            document_id = os.path.splitext(filename)[0].replace('PI_', '')  # Remove 'PI_' from document ID
            logging.info(f"Processing {filename}...")

            # Extract and preprocess text
            raw_text = extract_text_from_pdf(file_path)
            if raw_text is None:
                continue

            cleaned_text = remove_text_after_phrase(raw_text, "The information contained in this brochure")
            if not cleaned_text:
                logging.warning(f"Text removal resulted in empty text for {filename}. Skipping.")
                continue

            preprocessed_text = preprocess_text(cleaned_text)
            if preprocessed_text is None:
                continue

            # Encode text and document ID separately
            encoded_text = encode_text(preprocessed_text)
            encoded_id = encode_text(document_id)
            if encoded_text is None or encoded_id is None:
                continue

            # Normalize the embeddings
            normalized_text_embedding = normalize_vector(encoded_text)
            normalized_id_embedding = normalize_vector(encoded_id)

            # Store in MongoDB
            store_document_in_mongodb(document_id, preprocessed_text, normalized_text_embedding, normalized_id_embedding)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, '..', 'product_docs_us')
    folder_path = os.path.abspath(folder_path)
    process_documents(folder_path)
