# 🧠 Dense Retriever (SBERT) for Medical Question Answering

This project demonstrates a **dense retriever** using **Sentence-BERT** (`all-MiniLM-L6-v2`) to retrieve the most relevant answers to medical questions from the [MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD) dataset. It uses **Gradio** to provide an interactive user interface for semantic search.

---

## 🚀 Demo

Ask a medical question like:

> **"What are the genetic changes related to capillary malformation-arteriovenous malformation syndrome ?"**

The retriever returns the top most relevant answers from 2000 QA pairs in the MedQuAD dataset using dense vector similarity.

![image](https://github.com/user-attachments/assets/ccf74184-59e2-4966-9eb1-3b806c1902cb)


---

## 📦 Features

- 🔍 **Dense Retrieval** using Sentence-BERT (`all-MiniLM-L6-v2`)
- 🧬 **Medical QA** with MedQuAD dataset
- 🧠 Semantic similarity using **cosine similarity**
- 💻 **Interactive Gradio Interface**
- 💾 Embeddings caching using `joblib`
- 🧪 Easy to extend or fine-tune

---

## 🛠️ Installation

```bash
pip install sentence-transformers datasets gradio joblib torch
```

## 📊 Dataset

- **Name**: [MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD)

- **Size**: 47K QA pairs (used a subset of 2,000 for demo)

- **Fields**: `question`, `answer`

Load the dataset: I only used 2000 samples; you can use whatever numbers you want.

```python
from datasets import load_dataset

dataset = load_dataset("lavita/MedQuAD", split="train[:2000]")

```

## 🤖 Model

- [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

- 384-dimensional sentence embeddings

- Lightweight and efficient for semantic search


## 🔍 How It Works

**Precompute Embeddings** for each question+answer pair using SBERT.

**Store Embeddings** using `joblib` for fast loading.

On **user query**, compute embedding → cosine similarity → retrieve top-k.

Display using **Gradio** interface.










