import ollama, chromadb
client = chromadb.PersistentClient(path="./arxiv_vector_db")
collection = client.get_collection(name="arxiv_papers")
while True:
    q = input("You: ").strip()
    if not q: continue
    if q.lower() in ['exit', 'quit']: break
    print("Searching...")
    emb = ollama.embeddings(model="nomic-embed-text", prompt=q)['embedding']
    res = collection.query(query_embeddings=[emb], n_results=5)
    ctx = "\n\n".join(res['documents'][0])
    stream = ollama.chat(model='llama3.2', messages=[
        {'role': 'system', 'content': 'Use the context to answer. Stay concise.'},
        {'role': 'user', 'content': f"Context: {ctx}\n\nQuestion: {q}"}
    ], stream=True)
    print("\nAI: ", end="")
    for c in stream: print(c['message']['content'], end='', flush=True)
    print("\n\n")
