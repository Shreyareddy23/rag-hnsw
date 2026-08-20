# Advanced RAG System with HNSW

An advanced Retrieval-Augmented Generation (RAG) system that combines
Sentence Transformers, Qdrant vector search, HNSW approximate nearest
neighbor indexing, and Groq-hosted Large Language Models.

The system converts documents into semantic vector embeddings, stores them
in a Qdrant vector database indexed using HNSW, retrieves the most relevant
document chunks for a user query, and uses an LLM to generate a grounded
answer from the retrieved context.

The project also evaluates HNSW against exact vector search and analyzes the
effect of HNSW search parameters such as `ef` on retrieval latency and
Recall@K.

---

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [System Architecture](#system-architecture)
- [RAG Pipeline](#rag-pipeline)
- [HNSW](#hnsw)
- [How HNSW Works](#how-hnsw-works)
- [HNSW Search Process](#hnsw-search-process)
- [HNSW Parameters](#hnsw-parameters)
- [Why HNSW](#why-hnsw)
- [Project Workflow](#project-workflow)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Benchmarking](#benchmarking)
- [Evaluation Metrics](#evaluation-metrics)
- [Experimental Results](#experimental-results)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

# Overview

Retrieval-Augmented Generation (RAG) improves Large Language Model (LLM)
responses by providing relevant external information as context before
generating an answer.

Instead of relying only on knowledge stored inside the model, the system
first searches a document collection and retrieves relevant information.

The basic workflow is:

```text
User Query
    |
    v
Query Embedding
    |
    v
Vector Search
    |
    v
Relevant Document Chunks
    |
    v
Context + Query
    |
    v
LLM
    |
    v
Grounded Answer
