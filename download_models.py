from sentence_transformers import SentenceTransformer
import os

# Create a folder for models
model_path = "./models/all-MiniLM-L6-v2"
if not os.path.exists(model_path):
    print("Downloading Sentence Transformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.save(model_path)
    print("Done!")