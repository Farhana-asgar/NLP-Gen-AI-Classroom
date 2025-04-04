import pickle
from sentence_transformers import util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def load_pickled_embeddings(file_path="./saved_embeddings.pickle"):
    with open(file_path, "rb") as f:
        my_dict = pickle.load(f)
    return my_dict

word_vectors = load_pickled_embeddings()
cosine_sim={}
words = list(word_vectors.keys())
vectors = np.array(list(word_vectors.values()))  # Convert to NumPy array

# Compute cosine similarity using sentence-transformers
cos_sim_matrix = util.cos_sim(vectors, vectors).numpy()  # Convert to NumPy

# Convert to Pandas DataFrame for readability
df = pd.DataFrame(cos_sim_matrix, index=words, columns=words)

# Print the cosine similarity matrix
print(df)
df.to_csv("F:\Learning/GenAI/Assignment 1.1/cosine_similarity.csv")

plt.figure(figsize=(6,5))
sns.heatmap(df, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Cosine Similarity Matrix (Sentence Transformers)")
plt.show()