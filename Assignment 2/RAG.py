import ast
import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import torch
from langchain.chains import LLMChain
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from rank_bm25 import BM25Okapi
from reranker import Reranker
from retrieval_metrics import RetrievalMetrics
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfIdf:
    """
    TF-IDF based retriever that vectorizes context and retrieves top-k similar
    documents using cosine similarity.
    """

    def __init__(self):
        """
        Load Pretrained Models
        """
        model_path = "model/tf_idf/"
        with open(model_path + "tfidf_vectorizer.pkl", "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)
        with open(model_path + "context_vectors.pkl", "rb") as f:
            self.context_vectors = pickle.load(f)
        with open(model_path + "contexts.pkl", "rb") as f:
            self.contexts = pickle.load(f)

    def tfidf_retriever(self, query, top_k=3):
        """
        Finds the best matching contexts using cosine similarity

        Arguments:
        Query - The query
        Top_k - The number of top matching contexts to fetch

        Returns:
        List of Contexts Matched
        """
        try:
            query_vec = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(
                query_vec, self.context_vectors).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            return [self.contexts[i] for i in top_indices]
        except Exception as e:
            print(f"[Error in tfidf_retriever] {str(e)}")
            return []

    def train_tfidf_retriever(self, context_list):
        """
        Trains the model and stores it

        Arguments:
        contexts_list : List of contexts that needs to be trained

        """
        try:
            global tfidf_vectorizer, context_vectors, contexts
            print("\n=== Training TF-IDF Retriever ===")
            contexts = context_list
            tfidf_vectorizer = TfidfVectorizer()
            context_vectors = tfidf_vectorizer.fit_transform(contexts)
            model_path = "model/tf_idf/"
            with open(model_path + "tfidf_vectorizer.pkl", "wb") as f:
                pickle.dump(tfidf_vectorizer, f)
            with open(model_path + "context_vectors.pkl", "wb") as f:
                pickle.dump(context_vectors, f)
            with open(model_path + "contexts.pkl", "wb") as f:
                pickle.dump(contexts, f)
        except Exception as e:
            print(f"[Error in train_tfidf_retriever] {str(e)}")


class BM25:
    """
    BM25-based retriever that scores documents using bag-of-words
    matching with tokenized contexts.
    """

    def __init__(self):
        """
        Load Pretrained Models
        """
        model_path = "model/bm25/"
        with open(model_path + "bm25.pkl", "rb") as f:
            self.tokenized_contexts = pickle.load(f)
        self.bm25obj = BM25Okapi(self.tokenized_contexts)
        with open(model_path + "contexts.pkl", "rb") as f:
            self.contexts = pickle.load(f)

    def train_bm25_retriever(self, context_list):
        """
        Trains the model and stores it

        Arguments:
        contexts_list : List of contexts that needs to be trained

        """
        try:
            global bm25, contexts
            print("\n=== Training BM25 Retriever ===")
            contexts = context_list
            tokenized_contexts = [c.lower().split() for c in contexts]
            bm25 = BM25Okapi(tokenized_contexts)
            model_path = "model/bm25/"
            with open(model_path + "bm25.pkl", "wb") as f:
                pickle.dump(tokenized_contexts, f)
            with open(model_path + "contexts.pkl", "wb") as f:
                pickle.dump(contexts, f)
        except Exception as e:
            print(f"[Error in train_bm25_retriever] {str(e)}")

    def bm25_retriever(self, query, top_k=3):
        """
        Retrieve top-k documents using BM25 scores.

        Arguments:
        Query - The query
        Top_k - The number of top matching contexts to fetch

        Returns:
        List of Contexts Matched
        """
        try:
            tokenized_query = query.lower().split()
            scores = self.bm25obj.get_scores(tokenized_query)
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:top_k]
            return [self.contexts[i] for i in top_indices]
        except Exception as e:
            print(f"[Error in bm25_retriever] {str(e)}")
            return []


class VectorSearch:
    """
    Dense retrieval using pre-trained sentence embeddings from
    SentenceTransformer.
    """

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        model_path = "model/vector_search/"
        with open(model_path + "contexts.pkl", "rb") as f:
            self.contexts = pickle.load(f)
        self.context_embeddings = torch.load(model_path +
                                             "context_embeddings.pt")

    def train_vector_model(self, context_list):
        """
        Trains the model and stores it

        Arguments:
        contexts_list : List of contexts that needs to be trained

        """
        try:
            print("\n=== Training Vector Search Retriever ===")
            self.contexts = context_list
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.context_embeddings = self.model.encode(
                self.contexts, convert_to_tensor=True
            )
            model_path = "model/vector_search/"
            with open(model_path + "contexts.pkl", "wb") as f:
                pickle.dump(self.contexts, f)
            torch.save(self.context_embeddings, model_path +
                       "context_embeddings.pt")
        except Exception as e:
            print(f"[Error in train_vector_model] {str(e)}")
            return []

    def vector_search_retriever(self, query, top_k=3):
        """
        Retrieve top-k documents using cosine similarity of dense embeddings.

        Arguments:
        Query - The query
        Top_k - The number of top matching contexts to fetch

        Returns:
        List of Contexts Matched
        """
        try:
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(
                query_embedding, self.context_embeddings
            )[0]
            top_results = torch.topk(cosine_scores, k=top_k)
            return [self.contexts[idx] for idx in top_results.indices]
        except Exception as e:
            print(f"[Error in vector_retriever] {str(e)}")
            return []


class HybridSearch:
    """
    Hybrid retriever that combines BM25 and vector search scores.
    """

    def train_hybrid_model(self, context_list):
        """
        Trains the model and stores it

        Arguments:
        contexts_list : List of contexts that needs to be trained
        """
        try:
            print("\n=== Training Hybrid Retriever (BM25 + Vector) ===")
            self.contexts = context_list
            # Save contexts
            model_path = "model/hybrid/"
            with open(model_path + "contexts.pkl", "wb") as f:
                pickle.dump(self.contexts, f)

            # Train and save BM25
            tokenized_contexts = [c.lower().split() for c in self.contexts]
            with open(model_path + "bm25_tokenized.pkl", "wb") as f:
                pickle.dump(tokenized_contexts, f)

            # Train and save vector embeddings
            model = SentenceTransformer("all-MiniLM-L6-v2")
            context_embeddings = model.encode(self.contexts,
                                              convert_to_tensor=True)
            torch.save(context_embeddings, model_path +
                       "context_embeddings.pt")

            print("Hybrid model saved to model/hybrid/")
        except Exception as e:
            print(f"[Error in train_hybrid_model] {str(e)}")

    def hybrid_search_retriever(self, query, top_k=3, alpha=0.5):
        """
        Retrieve top-k documents by combining BM25 and vector scores
        with weight alpha.

        Arguments:
        Query - The query
        Top_k - The number of top matching contexts to fetch
        alpha = 0.0 => only BM25
                1.0 => only Vector
        """
        try:
            # Load contexts and tokenized BM25 data
            model_path = "model/hybrid/"
            with open(model_path + "contexts.pkl", "rb") as f:
                contexts = pickle.load(f)
            with open(model_path + "bm25_tokenized.pkl", "rb") as f:
                tokenized_contexts = pickle.load(f)

            # BM25 scoring
            bm25 = BM25Okapi(tokenized_contexts)
            bm25_scores = bm25.get_scores(query.lower().split())

            # Vector scoring
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = torch.load(model_path + "context_embeddings.pt")
            query_embedding = model.encode(query, convert_to_tensor=True)
            vector_scores = util.pytorch_cos_sim(query_embedding,
                                                 embeddings)[0].numpy()

            # Normalize and combine
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()
            combined = (
                alpha * scaler.fit_transform(
                    vector_scores.reshape(-1, 1)).flatten()
                + (1 - alpha)
                * scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
            )

            top_indices = combined.argsort()[-top_k:][::-1]
            return [contexts[i] for i in top_indices]
        except Exception as e:
            print(f"[Error in hybrid_search_retriever] {str(e)}")
            return []


class LLMPrediction:
    """
    Uses an LLM to answer questions given retrieved context.
    """

    def __init__(self):
        self.llm = Ollama(model="mistral")
        self.prompt = PromptTemplate.from_template(
            "You are an AI assistant. Use the following context to answer the \
                question. "
            "If you can't find the answer in the context, say 'I don’t know'. "
            "Question: {question}\nContext: {context}"
        )
        self.qa_chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def llm_prediction(self, retrieval_results):
        """
        Generate LLM answers for each question-context pair.
        Arguments:
        retrieval_results - Prepares the input and sends to LLM for prediction

        Returns:
        Predictions from LLMS
        """
        try:
            llm_answers = {}
            for top_k_key, retrievers_data in retrieval_results.items():
                llm_answers[top_k_key] = {}
                for retriever_name, outputs in retrievers_data.items():
                    print(f"Processing: {top_k_key} - {retriever_name}")
                    retriever_answers = []
                    input_data = []
                    # Prepare batches
                    for item in outputs:
                        temp_answers = []
                        question = item["question"]
                        context = "\n".join(item["contexts"])

                        # Create input for a single question-context pair
                        input_data.append({"question": question,
                                           "context": context})
                        print(item["answer"])
                        retriever_answers.append(
                            {
                                "question": item["question"],
                                "ground_truth": ast.literal_eval(
                                    item["answer"])["text"][0],
                                "contexts": item["contexts"],
                            }
                        )

                        # Generate prediction
                    prediction = self.qa_chain.apply(input_data)
                    print(f"Got Prediction for {top_k_key} - {retriever_name}")
                    for i in range(len(retriever_answers)):
                        retriever_answers[i]["predicted_answer"] = \
                            prediction[i]["text"]

                    llm_answers[top_k_key][retriever_name] = retriever_answers
            return llm_answers
        except Exception as e:
            print(f"[Error in llm_prediction] {str(e)}")
            return {}


def read_data(context, squad_v2_dataset):
    """
    Read context and QA dataset from JSONL files.

    Arguments:
    context - context.json file path
    squad_v2_dataset.json file path

    Retunrs
    Formatted Chunked Data
    """
    try:
        with open(context, "r") as f1, open(squad_v2_dataset, "r") as f2:
            json1 = [json.loads(line) for line in f1]
            json2 = [json.loads(line) for line in f2]
        return chunk_data(json1, json2)
    except Exception as e:
        print(f"[Error in read_data] {str(e)}")
        return []


def chunk_data(json1, json2):
    """
    Combine data from context and question-answer JSONs into a single list.

    Arguments:
    json1, json2 - JSON input to process

    Returns:
    Combined data
    """
    try:
        chunked_data = []
        for obj1, obj2 in zip(json1, json2):
            combined = obj1.copy()
            for key, value in obj2.items():
                combined[key] = value
            chunked_data.append(combined)
        return chunked_data
    except Exception as e:
        print(f"[Error in chunk_data] {str(e)}")
        return []


def retriever(chunked_data, retriever_methods):
    """
    Retrieve top-k documents using different retrievers and rerank them.
    Arguments:
    chunked_data - The chunked input data with question, answer and context
    retriever_methods - The different
    """
    try:
        questions = [item["question"] for item in chunked_data]
        answers = [item["answers"] for item in chunked_data]
        top_k_values = [3, 5]
        retrieval_results = {}
        retrieval_results_reranked = {}
        sampled_data = list(zip(questions, answers))[:5]
        reranker = Reranker()

        for top_k in top_k_values:
            print(f"\nRetrieving for top_k = {top_k}")
            top_k_key = f"top_k={top_k}"
            retrieval_results[top_k_key] = {}
            retrieval_results_reranked[top_k_key] = {}
            for name, retriever_func in retriever_methods.items():
                retriever_outputs = []
                retriever_outputs_reranked = []

                for q, a in sampled_data:
                    retrieved_contexts = retriever_func(q, top_k=top_k)
                    try:
                        reranked_contexts = reranker.rerank(
                            q, retrieved_contexts)
                    except Exception as e:
                        print(e)
                        reranked_contexts = retrieved_contexts
                    # reranked_contexts = reranker.rerank(q,
                    # retrieved_contexts)
                    # Reranking to enhance performance
                    retriever_outputs_reranked.append(
                        {"question": q, "answer": a,
                         "contexts": reranked_contexts}
                    )
                    retriever_outputs.append(
                        {"question": q, "answer": a,
                         "contexts": retrieved_contexts}
                    )
                retrieval_results_reranked[top_k_key][name] =\
                    retriever_outputs_reranked
                retrieval_results[top_k_key][name] = retriever_outputs
        return retrieval_results, retrieval_results_reranked
    except Exception as e:
        print(f"[Error in retriever] {str(e)}")
        return {}, {}


def plot_bar_chart(metrics):
    """
    Plots the data

    Arguments:
    metrics - The data to plot
    """
    try:
        df = pd.DataFrame(metrics).transpose()

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot on the specified axes
        df.plot(kind="bar", width=0.8, ax=ax)

        # Set plot labels and title
        ax.set_title("Retrieval Metrics Comparison for Different k Values")
        ax.set_xlabel("k Value")
        ax.set_ylabel("Metric Score")
        ax.set_ylim(0, 1)  # Assuming metric scores range between 0 and 1
        ax.set_xticklabels(df.index, rotation=0)
        ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc="upper left")

        # Adjust layout
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[Error in plot_bar_chart] {str(e)}")
        return []


def main():
    # Read data
    context = "F:/Learning/Personal GIT/NLP-Gen-AI-Classroom/Assignment 2/Dataset/Squad_v2_sampled/context.jsonl"
    squad_v2_dataset = "F:/Learning/Personal GIT/NLP-Gen-AI-Classroom/Assignment 2/Dataset/Squad_v2_sampled/squad_v2_dataset.jsonl"
    chunked_data = read_data(context, squad_v2_dataset)

    # Train the model
    # contexts = [item["context"] for item in chunked_data]
    tf_idf = TfIdf()
    bm25 = BM25()
    vector_search = VectorSearch()
    hybrid_search = HybridSearch()
    # tf_idf.train_tfidf_retriever(contexts)
    # bm25.train_bm25_retriever(contexts)
    # vector_search.train_vector_model(contexts)
    # hybrid_search.train_hybrid_model(contexts)

    # Retriever
    retriever_methods = {
        "TF-IDF": tf_idf.tfidf_retriever,
        "BM25": bm25.bm25_retriever,
        "Vector": vector_search.vector_search_retriever,
        "Hybrid": hybrid_search.hybrid_search_retriever,
    }
    retrieval_results, retrieval_results_reranked = retriever(
        chunked_data, retriever_methods
    )

    # Prediction
    llm_prediction = LLMPrediction()
    llm_answers = llm_prediction.llm_prediction(retrieval_results)
    llm_answers_reranked = llm_prediction.llm_prediction(
        retrieval_results_reranked)

    with open("output/llm_generated_answers_reranked.json", "w") as f:
        json.dump(llm_answers_reranked, f, indent=2)
    with open("output/llm_generated_answers.json", "w") as f:
        json.dump(llm_answers, f, indent=2)
    print("Predictions saved")
    # with open("output/llm_generated_answers_reranked.json", "rb") as f:
    #     llm_answers_reranked = json.load(f)
    # with open("output/llm_generated_answers.json", "rb") as f:
    #     llm_answers = json.load(f)
    # print("Loaded Response Data")

    # Metrics Calculation
    obj = RetrievalMetrics()
    print("Calculating Metrics before Reranking")
    average_metrics_result = obj.average_metrics(llm_answers=llm_answers)
    print("Calculating Metrics after Reranking")
    average_metrics_result_reranked = obj.average_metrics(
        llm_answers=llm_answers_reranked
    )

    # Plotting the metrics
    print("Plotting Graph")
    plot_bar_chart(average_metrics_result)
    plot_bar_chart(average_metrics_result_reranked)


main()
