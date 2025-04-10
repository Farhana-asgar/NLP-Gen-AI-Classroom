# file: cosine_similarity.py

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import util


def load_embeddings(file_path="./saved_embeddings.pickle"):
    """Load embeddings from a pickle file."""
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        raise


def compute_cosine_similarity(vectors, words):
    try:
        sim_matrix = util.cos_sim(vectors, vectors).numpy()
        return pd.DataFrame(sim_matrix, index=words, columns=words)
    except Exception as e:
        print(f"Error computing cosine similarity: {e}")
        raise


def save_to_csv(df, path):
    """Save DataFrame to CSV."""
    try:
        df.to_csv(path)
        print(f"Saved similarity matrix to {path}")
    except Exception as e:
        print(f"Error saving CSV to {path}: {e}")
        raise


def plot_heatmap(df, title="Cosine Similarity Matrix", figsize=(6, 5)):
    """Plot cosine similarity heatmap."""
    try:
        plt.figure(figsize=figsize)
        sns.heatmap(df, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title(title)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting heatmap: {e}")
        raise


def main(
    embedding_path="./saved_embeddings.pickle",
    output_csv="cosine_similarity.csv"
):
    try:
        word_vectors = load_embeddings(embedding_path)
        vectors = np.array(list(word_vectors.values()))
        words = list(word_vectors.keys())

        similarity_df = compute_cosine_similarity(vectors, words)
        save_to_csv(similarity_df, output_csv)
        plot_heatmap(similarity_df)
    except Exception as e:
        print(f"Execution failed: {e}")


if __name__ == "__main__":
    main()
