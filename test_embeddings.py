from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode("Hello world!")
print(emb[:5])
print(len(emb))
