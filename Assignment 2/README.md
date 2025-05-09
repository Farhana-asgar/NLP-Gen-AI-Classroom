# RAG

This project explores various RAG techniques available and evaluates the performance of a retrieval system with and without reranking, tested with different values of k. The evaluation uses standard retrieval metrics to analyze the quality of results.

## Artifacts
### Folders

1. Dataset - Has the RAG dataset to be used
2. Model - Has all the trained Models
3. Output - Prediction and metrics

### Files

1. RAG.py - The main file of the project
2. reranker.py - The reranker module
3. retrieval_metrics.py - Calculation of various metrics


## Execution

```bash
python RAG.py
```
## Overview
1. Models - TF-IDF, BM25, Vector Search, Hybrid
2. Reranker - BERT (amberoad/bert-multilingual-passage-reranking-msmarco)
3. Evaluation Metrics - Precision@k, Recall@k, F1@k, HitRate@k, MRR@k, nDCG@k, MAP

## License

[MIT](https://choosealicense.com/licenses/mit/)