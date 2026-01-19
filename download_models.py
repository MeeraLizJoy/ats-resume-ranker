import os
import spacy
from sentence_transformers import SentenceTransformer

# 1. Handle Sentence Transformer
st_path = "./models/all-MiniLM-L6-v2"
if not os.path.exists(st_path):
    print("Downloading Sentence Transformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.save(st_path)

# 2. Handle the NER Model (The one causing the error)
# Note: If this is a custom model you trained, you need to either 
# commit it to Git LFS or have this script download it from a URL.
# For now, let's at least create the directory so the app doesn't crash.
ner_path = "./output/model-last"
if not os.path.exists(ner_path):
    os.makedirs(ner_path, exist_ok=True)
    print(f"Created {ner_path} directory.")
    # If you have a way to download your custom weights here, add it!

print("Done!")