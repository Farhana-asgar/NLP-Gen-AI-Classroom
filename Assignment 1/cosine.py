# file: cosine_similarity.py

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import util


def load_embeddings(file_path="./saved_embeddings.pickle"):
    """
    Load word embeddings from a pickle file.

    Parameters
    ----------
    file_path : str, optional
        Path to the pickle file containing word embeddings,
        by default "./saved_embeddings.pickle".

    Returns
    -------
    dict
        Dictionary where keys are words and values are embedding vectors.

    Raises
    ------
    FileNotFoundError
        If the specified file is not found.
    Exception
        If any other error occurs during loading.
    """
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
    """
    Compute cosine similarity matrix for the given word vectors.

    Parameters
    ----------
    vectors : numpy.ndarray
        A NumPy array of shape (n_words, n_dimensions) representing word embeddings.
    words : list of str
        List of words corresponding to the vectors.

    Returns
    -------
    pandas.DataFrame
        A DataFrame representing the cosine similarity matrix,
        with words as both row and column labels.

    Raises
    ------
    Exception
        If cosine similarity computation fails.
    """
    try:
        sim_matrix = util.cos_sim(vectors, vectors).numpy()
        return pd.DataFrame(sim_matrix, index=words, columns=words)
    except Exception as e:
        print(f"Error computing cosine similarity: {e}")
        raise


def save_to_csv(df, path):
    """
    Save the similarity matrix DataFrame to a CSV file.

    Parameters
    ----------
    df : pandas.DataFrame
        The similarity matrix to be saved.
    path : str
        File path where the CSV should be written.

    Raises
    ------
    Exception
        If saving to CSV fails.
    """
    try:
        df.to_csv(path)
        print(f"Saved similarity matrix to {path}")
    except Exception as e:
        print(f"Error saving CSV to {path}: {e}")
        raise


def plot_heatmap(df, title="Cosine Similarity Matrix", figsize=(6, 5)):
    """
    Plot a heatmap of the cosine similarity matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        Cosine similarity matrix to be visualized.
    title : str, optional
        Title of the plot, by default "Cosine Similarity Matrix".
    figsize : tuple of int, optional
        Size of the figure, by default (6, 5).

    Raises
    ------
    Exception
        If heatmap plotting fails.
    """
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
    embedding_path="./saved_embeddings.pickle", output_csv="cosine_similarity.csv"
):
    """
    Main function to load embeddings, compute cosine similarity, save to CSV, and plot heatmap.

    Parameters
    ----------
    embedding_path : str, optional
        Path to the pickle file containing word embeddings,
        by default "./saved_embeddings.pickle".
    output_csv : str, optional
        Path where the cosine similarity matrix CSV will be saved,
        by default "cosine_similarity.csv".

    Raises
    ------
    Exception
        If any step in the pipeline fails.
    """
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
