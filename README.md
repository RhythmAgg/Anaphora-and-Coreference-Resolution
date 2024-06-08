# Anaphora and Coreference Resolution
This project aims to develop a coreference resolution system that identifies and links mentions referring to the same entity within a text document. The system will address the challenges of linguistic ambiguity, elided mentions, and discourse phenomena using a higher-order coreference resolution approach with a rule-based and a neural network architecture and evaluate its performance against state-of-the-art (SoTA) models. We will explore the effectiveness of our chosen neural architecture and compare its results with leading models on benchmark datasets.

## Datasets and Models
- `Dataset` : CoNLL 2012 ( English )
- `Deterministic Coreference Model` : Stanford’s Multi-Pass Sieve Coreference Resolution System at the CoNLL.
- `Neural Coreference Model` : Higher-order Coreference Resolution with Coarse-to-fine Inference. (e2e-coref) with Bert Encoder.
- `SOTA Model` : Coreference Resolution through a seq2seq Transition-Based System

## Literature Review
Our literature review will explore various approaches to coreference resolution, including:
- **Rule-based Systems**:
We will examine the approach proposed in ["Deterministic Coreference Resolution Based on Entity-Centric, Precision-Ranked Rules" by Lee et al. (2013)](https://aclanthology.org/J13-4004/). This paper details a rule-based system that utilizes a "sieve" architecture with multiple coreference models applied sequentially. Each model leverages entity-centric information and builds upon the previous model's output, aiming for high precision and recall.
- **Higher-Order Coreference Resolution**:
We will delve into ["Higher-order Coreference Resolution with Coarse-to-fine Inference" by Lee et al. (2018)](https://arxiv.org/abs/1804.05392). This paper proposes a neural approach that iteratively refines mention representations for more accurate coreference resolution. We will analyze its architecture, training process, and reported performance
- **Seq2Seq Coreference Resolution**:
We will thoroughly study ["Coreference Resolution through a seq2seq Transition-Based System" by Bohnet et al. (2023)](https://arxiv.org/abs/2211.12142). This paper presents a seq2seq model that directly predicts coreference annotations for a document. We will analyze their model architecture, training process, and achieved performance on benchmark datasets.

The detailed approaches for the deterministic and neural models can be further referred from the [Report](Report.pdf)