import json
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "secrets.env"))

# Load static config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Access patterns
DATA_DIR = config.get("data_dir", "data/")
TRANSCRIPT_ZIP_URL = config.get("transcript_zip_url")
VECTOR_STORE = config.get("vector_store", "FAISS")
RETRIEVAL_METHOD = config.get("retrieval_method", "with_neighbors")
# Secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
