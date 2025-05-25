# KCC Q&A Assistant


KCC Query Assistant - Offline AI Application

## Project Title: KCC Agriculture Q&A Assistant [Query Assistnat]

Author:
DHANYATHA S

---

Overview

The KCC Query Assistant is an offline-capable, local-first AI application designed to provide accurate agricultural advice based on real-world data from the Kisan Call Center (KCC). It combines retrieval-augmented generation (RAG), local language models (LLMs), and a streamlined user interface to empower farmers and agricultural advisors—especially in low-connectivity environments.




Problem Statement

To build an offline Q&A system that:

Uses the KCC dataset as its core knowledge base.

Supports local language models .

Retrieves answers based on semantic similarity.

Works fully offline but supports fallback Internet search.




>[!NOTE] 
>Insted of Internet fallback this system has fallback Knowledge based quries and responses curated with semantic search.

## Objectives

### 1. Data Integration & Preprocessing

- Downloaded KCC data from data.gov.in.

- Filtered the dataset for the Karnataka region.

- Cleaned and normalized data into consistent Q&A format.

- Exported both raw and preprocessed files (CSV format).


### 2. Local LLM Deployment

- Used local models  google/flan-t5-small and base model

- Quantized the models for CPU/GPU efficiency used pyTorch.

- Fully offline inference enabled for user queries.


### 3. Retrieval-Augmented Generation (RAG)

- Used sentence-transformers to encode question-answer pairs (used HuggingFcae Model: all-MiniLM-L6-v2).

- Clustered and categorized Q&A pairs.

- Stored embeddings in a FAISS vector database.

- Implemented semantic search to fetch top-k relevant answers.

- Built a fallback response system using clustered categories if context fails.


### 4. User Interface

- Built with Streamlit for simplicity and local hosting.

Features:

- Text input for queries.

- Model selection (small/base).

- Control sliders for result count and similarity threshold.

- Display of structured answers with Confidence level.




---

Application Flow Summary
```
Data Ingestion
  ⇒ Load raw KCC CSV
Preprocessing
  ⇒ Clean, normalize, and chunk into Q&A pairs
Embedding Generation
  ⇒ Generate embeddings using sentence-transformers
Vector Store Ingestion
  ⇒ Store in FAISS vector database
Query Handling
  ⇒ User enters query
    └─ If context found:
        └─ Pass to LLM and display result
    └─ Else:
        └─ Display fallback clustered knowledge 
```

---

UI Explanation (Ref: Attached Screenshot)
![image](https://github.com/user-attachments/assets/75646c14-54a3-42f3-a4ce-582ca6c655d9)

![image](https://github.com/user-attachments/assets/d3bec7d0-9365-49af-a581-362f47d4228e)

#### Users enter a natural-language query.
#### Semantic search is triggered in the background.
#### Results are shown clearly:
#### Top K answers retrieved from KCC database.
#### If matched, passed through LLM for enhancement.
#### If not matched, fallback suggestions from clustered data appear.
#### Parameters for model type and search tuning are available.



---
```
Sample Queries (Karnataka-centric)

1. How to treat pests in cotton organically?
2. What fertilizer is suitable for paddy in rainfed areas?
3. How to manage drought stress in groundnut?
4. What pesticide is recommended for chili crops?
5. Methods to increase yield in sugarcane?
6. What time to sow ragi in Karnataka?
7. Disease control in banana plantation?
8. How to prepare organic compost?
9. What are the best irrigation techniques for tomato?
10. Soil preparation steps for sunflower?

```


---

### Technical Stack

Language Models: Google Flan-T5 Small/Base

Embeddings: MPNet (via Sentence Transformers)

Vector Store: FAISS

Frontend: Streamlit

Offline Support: Fully local model + FAISS vector DB

---

Installation & Launch

Requirements
```
Python 3.10+
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.2
langchain>=0.1.0
langchain-community>=0.0.20
langchain-huggingface>=0.0.3

# Vector Database
faiss-cpu>=1.7.4

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Web Framework
streamlit>=1.28.0
streamlit-chat>=0.1.1

# Utilities
python-dotenv>=1.0.0


# Optional: GPU support (uncomment if you have CUDA)
# torch>=2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
# faiss-gpu>=1.7.4
```

Steps

1. Clone the repo:

```
git clone https://github.com/yourname/KCCQAAssistant.git
cd KCCQAAssistant
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the local model using Ollama:
```
ollama run flan-t5
or
Use models directly onto code
```
4. Launch the Streamlit app:
```
streamlit run kcc_app.py
```

---

Demonstration Script (1–2 Minutes)

"Hello everyone, I’m Dhanyatha S, and this is my project — the KCC Agricultural Q&A Assistant. This is an offline application that helps farmers get instant answers to agriculture-related queries, even without internet access.I downloaded real KCC advisory data from data.gov.in, filtered for Karnataka, and cleaned and structured the Q&A format.Next, I generated embeddings for each question-answer pair and stored them in a FAISS vector database.I also clustered similar Q&A pairs and integrated a local LLM using Google’s Flan-T5-Small via Ollama for completely offline generation.
This is the Streamlit UI. The user enters a query, and the system retrieves top semantic matches. If relevant context is found, it passes it to the LLM and shows a response. If not, it displays fallback answers from clustered queries.
This combines real data, AI, and a simple interface to help farmers access expert advice anytime."


---

Repository Structure
```
KCCQueryAssistant/
├── data/
│   ├── raw_kcc_data.csv
│   └── processed_kcc_data.csv
├── embeddings/
│   └── vector_store.py
├── models/
│   └── offline_llm.py
├── app.py
├── utils/
│   ├── preprocess.py
│   ├── embedding.py
│   
├── requirements.txt
└── README.md

```
---

Demo Video

Google Drive Link to Screencast (View-Only)

[Demo Video](https://drive.google.com/file/d/1dy2uxj3De0sjTQJmfZYzzzDNkLVqg6xk/view?usp=sharing)
---

Contact

For questions, reach out to: dhanyatha237.y@gmail.com or GitHub Profile: https://github.com/Dhanyatha-s
