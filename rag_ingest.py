import os, json, ollama, chromadb
from tqdm import tqdm
DATA_PATH, DB_PATH = "INSERT_HERE", "./arxiv_vector_db"
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="arxiv_papers")
if not os.path.exists(DATA_PATH): exit()
files = [f for f in os.listdir(DATA_PATH) if f.endswith('.json')]
for filename in tqdm(files):
    try:
        with open(os.path.join(DATA_PATH, filename), 'r') as f:
            data = json.load(f)
            text = data.get('text', "")
            if not text or len(collection.get(ids=[f"{filename}_0"])['ids']) > 0: continue
            chunks = [text[i:i+1200] for i in range(0, len(text), 1200)]
            for i, chunk in enumerate(chunks):
                emb = ollama.embeddings(model="nomic-embed-text", prompt=chunk)['embedding']
                collection.add(ids=[f"{filename}_{i}"], embeddings=[emb], documents=[chunk])
    except: continue
print("Done.")
