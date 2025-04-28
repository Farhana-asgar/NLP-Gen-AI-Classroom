import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Reranker:
    """
    Reranker class that uses a transformer-based model to rerank a
    list of contexts based on their relevance to a given query.
    It utilizes a sequence classification model
    to score each (query, context) pair and sorts them by score.
    """
    def __init__(
            self, model_name="amberoad/bert-multilingual-passage-reranking-msmarco"):
        """
        Initializes the Reranker with the specified pre-trained model.

        Args:
            model_name (str): The name of the Hugging Face model to be used
            for reranking.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name)

    def rerank(self, query, contexts):
        """
        Reranks the provided list of contexts based on their relevance to the
        input query.

        Args:
            query (str): The user's input query.
            contexts (List[str]): A list of retrieved contexts to be reranked.

        Returns:
            List[str]: The input contexts sorted in descending order of
            relevance to the query.
        """
        inputs = self.tokenizer(
            [(query, context) for context in contexts],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        scores = logits[:, 1].tolist()
        return [contexts[i] for i in sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True)]
