# Bibliography: Two-Phase Entity Resolution (Vector Blocking & Probabilistic Linkage)

This bibliography presents key academic papers, tools, and case studies that describe, analyze, or facilitate a **two-phase entity resolution pipeline** using:
1. **Vector search / dense embeddings** for fast, semantic candidate generation (blocking).
2. **Probabilistic models (Fellegi-Sunter / Splink)** for high-precision, interpretable matching/linkage confirmation.

---

## 1. Foundational Probabilistic Linkage & Splink

These works form the basis of the second phase of the pipeline, providing the mathematical framework for calculating match weights and probabilities.

*   **Fellegi, I. P., & Sunter, A. B. (1969).** *A Theory for Record Linkage.* Journal of the American Statistical Association, 64(328), 1183–1210.
    *   **Core Contribution:** The seminal mathematical framework for probabilistic record linkage. It defines how to calculate match probabilities ($m$-probabilities and $u$-probabilities) based on agreement/disagreement patterns across attributes. This is the exact statistical engine underlying tools like Splink.
*   **Linacre, R., et al. (2022).** *Splink: Fast, unsupervised probabilistic record linkage at scale.* Ministry of Justice UK.
    *   **Core Contribution:** Introduces Splink, a widely-used Python package for large-scale record linkage. It details how the expectation-maximization (EM) algorithm is used to estimate Fellegi-Sunter model parameters unsupervised, scaling via backend engines like DuckDB, Spark, and AWS Athena.
    *   **Repository:** [moj-analytical-services/splink](https://github.com/moj-analytical-services/splink)

---

## 2. Vector-Based Blocking & Dense Retrieval (Stage 1)

These papers discuss the first phase of the pipeline: utilizing neural embeddings and Approximate Nearest Neighbor (ANN) search to filter down a quadratic ($O(n^2)$) comparison space into high-recall candidate pairs.

*   **Wang, T., Lin, H., Han, X., Chen, X., Cao, B., & Sun, L. (2024).** *Towards Universal Dense Blocking for Entity Resolution.* arXiv preprint arXiv:2404.14811. (UniBlocker)
    *   **Core Contribution:** Introduces **UniBlocker**, a universal dense blocking framework that uses self-supervised contrastive learning pre-trained on domain-independent tabular data. This solves a major limitation of older dense blocking models, which required costly domain-specific training or labeled data to initialize vector embeddings.
    *   **Repository:** [tshu-w/uniblocker](https://github.com/tshu-w/uniblocker)
*   **Zeakis, A., Papadakis, G., Skoutas, D., & Koubarakis, M. (2023).** *Pre-trained Embeddings for Entity Resolution: An Experimental Analysis.* Proceedings of the VLDB Endowment (PVLDB), 16(11), 3239–3251.
    *   **Core Contribution:** Provides a comprehensive empirical analysis of 12 pre-trained transformer-based language models across 17 entity resolution benchmark datasets. It assesses performance in both the blocking (retrieval) and matching (reranking) phases, offering a definitive guide on efficiency-recall trade-offs.
*   **Ebraheem, M., Thuné, M., Chassanis, D., Stonebraker, M., Hofman, A., & Rekatsinas, T. (2018).** *Distributed Representations for Entity Resolution.* PVLDB, 11(11), 1454-1467. (DeepER)
    *   **Core Contribution:** One of the earliest architectures (DeepER) to represent entire record tuples as dense vectors using LSTMs, performing semantic blocking via Locality Sensitive Hashing (LSH) over these distributed representations.

---

## 3. Hybrid Two-Phase Pipelines (Vector Search + Classifiers/Linkage)

These papers and libraries describe the integration of the two stages, specifically bridging Approximate Nearest Neighbor (ANN) search with subsequent record linkage confirmation.

*   **Strojny, T., & Beręsewicz, M. (2026).** *BlockingPy: A Python package for blocking in record linkage.* SoftwareX, 34, 102583.
    *   **Core Contribution:** Introduces **BlockingPy**, a package built specifically to integrate vector-based blocking (utilizing FAISS and Approximate Nearest Neighbor search on CPU/GPU) with subsequent record linkage packages. The authors specifically outline integration paths with **Splink** for executing the Fellegi-Sunter scoring stage on the candidate pairs generated via dense search.
    *   **Repository:** [beresewicz/blockingpy](https://github.com/ncn-foreigners/BlockingPy)
*   **Brunner, U., & Stockinger, K. (2020).** *Entity Resolution using Pre-trained Embeddings.* arXiv preprint arXiv:2006.01258.
    *   **Core Contribution:** Outlines a two-phase pipeline where dense embeddings and cosine similarity are first used to retrieve potential candidate matches (optimizing recall), which are then evaluated by traditional classifiers/linkage models in the second phase to maximize precision.
*   **Tu, Y., et al. (2023).** *Integrating Pre-trained Language Models and Probabilistic Record Linkage.* Journal of Official Statistics.
    *   **Core Contribution:** Evaluates techniques to inject dense semantic embedding similarity scores directly as features into classical Fellegi-Sunter probabilistic record linkage models. This allows practitioners to benefit from transformer-based semantic understanding while retaining the statistical interpretability of the Fellegi-Sunter model.
