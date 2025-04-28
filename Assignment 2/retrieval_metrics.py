import ast
import math

import numpy as np


class RetrievalMetrics:
    def __init__(self, k=3):
        self.k = k

    def _normalize_text(self, text):
        return text.lower().strip()

    def _is_relevant(self, retrieved, ground_truths):
        return any(self._normalize_text(gt) in self._normalize_text(retrieved)
                   for gt in ground_truths)

    def _binary_relevance_vector(self, retrieved_contexts,
                                 ground_truth_answers):
        return [
            1 if self._is_relevant(ctx, ground_truth_answers) else 0
            for ctx in retrieved_contexts
        ]

    def precision_at_k(self, binary_vec):
        return np.sum(binary_vec[:self.k]) / self.k

    def recall_at_k(self, binary_vec, total_relevant=1):
        return np.sum(binary_vec[:self.k]) / total_relevant \
            if total_relevant > 0 else 0.0

    def f1_at_k(self, precision, recall):
        return (2 * precision * recall) / (precision + recall) \
            if (precision + recall) > 0 else 0.0

    def hit_rate_at_k(self, binary_vec):
        return 1.0 if np.sum(binary_vec[:self.k]) > 0 else 0.0

    def mrr_at_k(self, binary_vec):
        for i in range(min(self.k, len(binary_vec))):
            if binary_vec[i] == 1:
                return 1.0 / (i + 1)
        return 0.0

    def dcg_at_k(self, binary_vec):
        return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(
            binary_vec[:self.k]))

    def idcg_at_k(self, binary_vec):
        sorted_rels = sorted(binary_vec, reverse=True)
        return self.dcg_at_k(sorted_rels)

    def ndcg_at_k(self, binary_vec):
        dcg = self.dcg_at_k(binary_vec)
        idcg = self.idcg_at_k(binary_vec)
        return dcg / idcg if idcg > 0 else 0.0

    def mean_avg_precision(self, binary_vec):
        hits = 0
        sum_precisions = 0.0
        for i, rel in enumerate(binary_vec):
            if rel == 1:
                hits += 1
                sum_precisions += hits / (i + 1)
        return sum_precisions / hits if hits > 0 else 0.0

    def evaluate(self, retrieved_contexts, ground_truth_answers):
        binary_vec = self._binary_relevance_vector(retrieved_contexts,
                                                   ground_truth_answers)
        precision = self.precision_at_k(binary_vec)
        recall = self.recall_at_k(binary_vec)
        f1 = self.f1_at_k(precision, recall)
        hit_rate = self.hit_rate_at_k(binary_vec)
        mrr = self.mrr_at_k(binary_vec)
        ndcg = self.ndcg_at_k(binary_vec)
        map_score = self.mean_avg_precision(binary_vec)

        return {
            "Precision@k": round(precision, 4),
            "Recall@k": round(recall, 4),
            "F1@k": round(f1, 4),
            "HitRate@k": round(hit_rate, 4),
            "MRR@k": round(mrr, 4),
            "nDCG@k": round(ndcg, 4),
            "MAP": round(map_score, 4)
        }

    def average_metrics(self, llm_answers):
        averaged_metrics_result = {}
        for top_k_key, methods in llm_answers.items():
            k = int(top_k_key.split('=')[1])
            metrics_calculator = RetrievalMetrics(k=k)
            aggregate_metrics = {
                "Precision@k": [],
                "Recall@k": [],
                "F1@k": [],
                "HitRate@k": [],
                "MRR@k": [],
                "nDCG@k": [],
                "MAP": []
            }

            # Iterate through each method and its entries
            for method_name, entries in methods.items():
                for entry in entries:
                    ground_truth_str = entry['ground_truth']
                    # Safely evaluate the string to a dictionary
                    ground_truth = ast.literal_eval(ground_truth_str)
                    ground_truth_texts = ground_truth['text']
                    contexts = entry['contexts']

                    # Evaluate metrics
                    scores = metrics_calculator.evaluate(contexts,
                                                         ground_truth_texts)

                    # Append each metric to the aggregate lists
                    for metric_name, value in scores.items():
                        aggregate_metrics[metric_name].append(value)

            # Compute the average for each metric
            averaged_metrics = {metric: round(sum(values) / len(values), 4)
                                if values else 0.0
                                for metric, values in aggregate_metrics.items()
                                }

            # Store the averaged metrics in the results dictionary
            averaged_metrics_result[f"k={k}"] = averaged_metrics
        return averaged_metrics_result