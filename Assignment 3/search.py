from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
query = "What is Widget A made of?"
query_embedding = model.encode([query])
D, I = index.search(np.array(query_embedding), k=3)

results = [metadata[i] for i in I[0]]
