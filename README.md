# Contextual Representation Geometry in Transformer Language Models

An empirical study of how contextual representations evolve across transformer layers using encoder and decoder language models.

---

## Overview

This project investigates how transformer language models represent **polysemous words** (words with multiple meanings) across their hidden layers.

Rather than evaluating downstream task performance, this work focuses on the **geometry of contextual embeddings**, analyzing how lexical meaning emerges throughout the network.

The study compares both encoder and decoder architectures using identical datasets and evaluation metrics.

Models evaluated:

- BERT
- RoBERTa
- GPT-2
- Llama 3.2 (1B)
- Qwen 2.5

---

## Research Question

How do transformer architectures differ in the way they represent contextual lexical semantics across their hidden layers?

Specifically, we investigate:

- How contextual representations evolve through depth
- When lexical senses become distinguishable
- How embedding geometry changes during contextualization
- Differences between encoder and decoder architectures

---

## Dataset

The experiments use the **SemCor** corpus from NLTK.

Target polysemous words include:

- bank
- plant
- spring
- bat
- crane
- seal

Each occurrence is extracted together with its sentence context.

---

## Methodology
<img width="1536" height="1024" alt="ChatGPT Image 24 juin 2026, 03_35_43" src="https://github.com/user-attachments/assets/7472573b-c4cc-4b86-90e4-1a8c8cdad797" />


For every occurrence of each target word:

1. Extract contextual embeddings from every transformer layer.
2. Compute multiple geometric metrics.
3. Compare representations across models.

The evaluation pipeline consists of:

- Context extraction from SemCor
- Hidden state extraction
- Token alignment
- Layer-wise embedding analysis
- Visualization
- Cross-model comparison

---

## Metrics

The project evaluates several complementary properties of contextual representations.

### Self-Similarity

Measures how similarly a word is represented across different contexts.

Lower similarity indicates stronger contextualization.

---

### Adjusted Self-Similarity

Corrects self-similarity by accounting for anisotropy.

---

### Maximum Explainable Variance (MEV)

Measures how concentrated contextual embeddings are around dominant directions.

---

### Anisotropy

Measures the geometric concentration of embedding vectors.

Higher anisotropy indicates embeddings occupy fewer directions in the representation space.

---

### K-Means Clustering

Clusters contextual embeddings according to lexical sense.

Used to evaluate semantic separability.

---

### Silhouette Score

Quantifies cluster quality.

Higher values indicate better lexical sense separation.

---

### Semantic Onset Depth

Estimates the transformer layer where contextual semantic distinctions first emerge.

---

## Models

| Model | Architecture |
|--------|--------------|
| BERT | Encoder |
| RoBERTa | Encoder |
| GPT-2 | Decoder |
| Llama 3.2 | Decoder |
| Qwen 2.5 | Decoder |

---

## Repository Structure

```
.
├── data/
│   ├── bank_contexts.json
│   ├── plant_contexts.json
│   └── ...
│
├── notebooks/
│   ├── bert_analysis.ipynb
│   ├── roberta_analysis.ipynb
│   ├── gpt2_analysis.ipynb
│   ├── llama_analysis.ipynb
│   └── qwen_analysis.ipynb
│
├── src/
│   ├── extraction.py
│   ├── embedding.py
│   ├── metrics.py
│   ├── clustering.py
│   └── visualization.py
│
├── report/
│   └── report.pdf
│
└── README.md
```

---

## Main Findings

The experiments reveal clear differences between encoder and decoder transformer architectures.

### Encoder Models

- Strong contextual specialization
- Better lexical sense separation
- Earlier semantic onset
- Lower anisotropy in deeper layers

### Decoder Models

- More stable contextual representations
- Later emergence of semantic distinctions
- Higher anisotropy
- Less distinct lexical sense clustering

Despite achieving comparable downstream performance, different transformer architectures organize semantic information using noticeably different internal geometries.

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- NumPy
- Scikit-learn
- Matplotlib
- NLTK

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/contextual-representation-geometry.git

cd contextual-representation-geometry
```

Install dependencies

```bash
pip install -r requirements.txt
```

Download SemCor

```python
import nltk

nltk.download("semcor")
```

---

## Running the Experiments

Run the notebooks corresponding to each model.

Example:

```
notebooks/
    bert_analysis.ipynb
```

Each notebook performs:

- context extraction
- hidden state extraction
- metric computation
- visualization
- clustering
- analysis

---

## Report

A complete research report describing the methodology, experiments, results, and discussion is available in

```
llm_analysis.pdf
```

---

## Future Work

Possible extensions include:

- Larger lexical evaluation sets
- Multilingual analysis
- More recent open-weight LLMs
- Statistical significance testing
- Additional geometric metrics
- Cross-lingual contextual representation analysis

---



## Author

**Benalia Meriem**

Machine Learning • Natural Language Processing • Representation Learning

GitHub: https://github.com/bn_meghiem
LinkedIn: https://linkedin.com/in/Benalia Meriem

---
