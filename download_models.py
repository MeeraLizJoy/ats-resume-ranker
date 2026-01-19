import os
from sentence_transformers import SentenceTransformer

# 1. Create the folders your app expects
os.makedirs("./models/all-MiniLM-L6-v2", exist_ok=True)
os.makedirs("./output/model-last", exist_ok=True)

# 2. Download the Sentence Transformer
print("Downloading Sentence Transformer...")
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save("./models/all-MiniLM-L6-v2")

# 3. Create a dummy meta.json if it's missing (to stop the Spacy E053 error)
# Note: You will eventually need to upload your real NER model files here
meta_path = "./output/model-last/meta.json"
if not os.path.exists(meta_path):
    with open(meta_path, "w") as f:
        f.write('{"lang":"en", "name":"last", "version":"0.0.0"}')

print("Done!")