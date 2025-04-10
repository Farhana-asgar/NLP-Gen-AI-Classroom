# file: cosine_similarity.py

import pickle
import numpy as np
import pandas as pd
from sentence_transformers import util
import matplotlib.pyplot as plt
import seaborn as sns


def load_embeddings(file_path="./saved_embeddings.pickle"):
    """Load embeddings from a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def compute_cosine_similarity(embedding_dict):
    """Compute cosine similarity matrix using sentence-transformers."""
    vectors = np.array(list(embedding_dict.values()))
    words = list(embedding_dict.keys())
    sim_matrix = util.cos_sim(vectors, vectors).numpy()
    return pd.DataFrame(sim_matrix, index=words, columns=words)


def save_to_csv(df, path):
    """Save DataFrame to CSV."""
    df.to_csv(path)


def plot_heatmap(df, title="Cosine Similarity Matrix", figsize=(6, 5)):
    """Plot cosine similarity heatmap."""
    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main(
    embedding_path="./saved_embeddings.pickle",
    output_csv="cosine_similarity.csv"
):
    word_vectors = load_embeddings(embedding_path)
    similarity_df = compute_cosine_similarity(word_vectors)
    save_to_csv(similarity_df, output_csv)
    plot_heatmap(similarity_df)


if __name__ == "__main__":
    main()
