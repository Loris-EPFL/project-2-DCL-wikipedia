from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("intfloat/multilingual-e5-large")

v1 = model.encode("Paris is the capital of France.")
v2 = model.encode("I like to eat bananas.")

print(np.linalg.norm(v1 - v2))
