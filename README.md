# Document Retrieval and Q&A System with RAG Workflow

## Overview

This project provides a Retrieval-Augmented Generation (RAG) workflow for building a document retrieval and Q&A system. The pipeline includes extracting text from PDFs, preprocessing and embedding data, storing documents in MongoDB, and deploying a chatbot interface using OpenAI's GPT models for natural language responses based on retrieved documents.

---

## Features
- **Data Processing**:
  - Extract text from PDFs using PyMuPDF (`fitz`).
  - Preprocess text with tokenization, stopword removal, and lemmatization.
  - Generate embeddings for documents and IDs using Sentence Transformers.

- **Database Integration**:
  - Store and retrieve processed documents and embeddings in MongoDB.

- **Document Retrieval**:
  - Perform similarity-based retrieval using cosine similarity.
  - Rank and return the top relevant documents for a query.

- **Chatbot Interface**:
  - Generate responses based on retrieved documents using OpenAI GPT models.
  - Provide a user-friendly interface via Gradio.

---

## Requirements

### Prerequisites
- Python 3.8 or higher
- MongoDB installed locally or accessible via URI
- OpenAI API key for GPT integration
- Required Python libraries (listed in `requirements.txt`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/document-retrieval-rag.git
   cd document-retrieval-rag
