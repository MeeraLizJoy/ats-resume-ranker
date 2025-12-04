import os
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

jd_folder = 'data/job_descriptions/'
out_folder = 'models/'

for filename in os.listdir(jd_folder):
    if filename.endswith('.txt'):
        with open(os.path.join(jd_folder, filename), 'r', encoding='utf-8') as f:
            text = f.read()
        embedding = model.encode([text])[0]
        out_path = os.path.join(out_folder, filename.replace('.txt', '_embedding.npy'))
        np.save(out_path, embedding)
        print(f"Saved embedding for {filename} to {out_path}")
