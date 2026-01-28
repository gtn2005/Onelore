# OneLore 

OneLore is a **knowledge-based AI chatbot** designed to answer questions about *One Piece* lore using **retrieval-based NLP** techniques. Instead of relying on memorized responses, OneLore fetches relevant information from a structured knowledge base and generates accurate answers using semantic similarity and transformer models.

---

##  Project Overview

The system is built as a **Retrieval-Augmented Question Answering (QA)** pipeline:

* Lore data is **fetched from the web**, cleaned, and structured into JSON
* Text is converted into **sentence embeddings**
* User queries are matched using **cosine similarity**
* A transformer-based QA model extracts precise answers from retrieved context

This approach ensures answers are grounded in data and remain explainable.

---

##  How It Works (Architecture)

1. **Data Collection & Cleaning**

   * One Piece lore is gathered from online sources
   * Irrelevant content is removed
   * Cleaned data is stored in a structured `lore_data.json` file

2. **Text Chunking & Embedding**

   * Lore entries are split into overlapping chunks
   * Each chunk is embedded using `all-MiniLM-L6-v2`
   * Embeddings are stored locally in `embeddings.pkl`

3. **Semantic Retrieval**

   * User queries are embedded using the same model
   * Cosine similarity is used to retrieve the most relevant lore chunks

4. **Question Answering**

   * Retrieved context is passed to a transformer QA model (`roberta-base-squad2`)
   * The model extracts a concise answer from the context

5. **Interactive UI**

   * Built using **Streamlit**
   * Displays answers along with their source lore entries

---

##  Tech Stack

* **Python**
* **Streamlit** – UI
* **Sentence Transformers** – Embeddings
* **Cosine Similarity (scikit-learn)** – Retrieval
* **Hugging Face Transformers** – Question Answering
* **JSON / Pickle** – Data storage

---

##  Project Structure

```
OneLore/
│── app.py               # Streamlit application
│── embed.py             # Data cleaning, chunking, and embedding
│── retrieve.py          # Cosine similarity-based retrieval
│── lore_data.json       # Cleaned and structured lore data
│── embeddings.pkl       # Precomputed embeddings
│── reset_knowledge.py   # Utility to regenerate embeddings
│── README.md            # Project documentation
```

---

##   Getting Started

###  Install Dependencies

```bash
pip install streamlit sentence-transformers transformers scikit-learn
```

###  Generate Embeddings

```bash
python embed.py
```

###  Run the App

```bash
streamlit run app.py
```


* Expand lore dataset
* Add citation confidence scores
* Replace QA model with a lightweight LLM
* Deploy on cloud (Hugging Face / Streamlit Cloud)

---

 Disclaimer

This project is for educational purposes only. *One Piece* and its characters are the property of Eiichiro Oda and associated publishers.

---


