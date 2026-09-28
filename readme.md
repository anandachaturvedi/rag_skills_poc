# Skills + RAG PoC

A simple Proof of Concept demonstrating how **LLM-based Skills and Retrieval-Augmented Generation (RAG)** can be combined to reduce the amount of context sent to an LLM.

## Overview

This project demonstrates a pipeline where:

1. An LLM selects the most relevant Skill based on the user's question.
2. The selected Skill points to its corresponding knowledge source.
3. The knowledge source is split into overlapping chunks.
4. Each chunk is converted into an embedding using `all-MiniLM-L6-v2`.
5. The user's question is also converted into an embedding.
6. Cosine similarity is used to find the most relevant chunks.
7. The top 2 chunks are retrieved.
8. Only the retrieved context is sent to the final LLM to generate the answer.

## Architecture

User Question
      ↓
LLM-based Skill Selection
      ↓
Selected Skill
      ↓
Knowledge Source
      ↓
Document Chunking
      ↓
Chunk Embeddings
      ↓
Query Embedding
      ↓
Cosine Similarity
      ↓
Top 2 Relevant Chunks
      ↓
Final LLM
      ↓
    Answer

**Project Structure**
    
rag_skills_poc/
│
├── documents/
│   ├── clockmakers_garden.txt
│   ├── deep_purple.txt
│   └── last_lighthouse.txt
│
├── skills/
│   ├── clockmakers_garden/
│   │   └── skills.md
│   ├── last_lighthouse/
│   │   └── skills.md
│   └── purple_apple/
│       └── skills.md
│
├── poc_skills.py
├── readme.md
└── .gitignore

**Skills**

Each Skill contains metadata describing when it should be selected and which knowledge source it uses.

Example:

---
name: the-deep-purple-apple
description: >-
  Use when the user's question involves the apple, its color,
  the orchard, Desmond, or the story The Deep Purple.
knowledge_source: documents/deep_purple.txt
---

The Skill is selected by the LLM, rather than using hardcoded keyword-based routing.

**RAG Retrieval**

The project uses the **all-MiniLM-L6-v2** Sentence Transformer model to generate embeddings.

Each document is divided into overlapping chunks.

The query and document chunks are represented as 384-dimensional vectors.

Cosine similarity is then used to determine which chunks are semantically closest to the user's question.

The top 2 chunks are provided to the final LLM as context.

**Example Question**
What color was the apple in the story?
Skill Selection
the-deep-purple-apple
Knowledge Source
documents/deep_purple.txt
Retrieved Context

The most relevant chunks containing information about the apple are retrieved.

**Final Answer**
The apple was purple.
Token Usage Comparison

The PoC was also used to compare token usage across different approaches.

Approach	Approx. Total Tokens
Full-context baseline	~2918
Skills-only	~1688
Skills + RAG	~545–604

These values are from PoC runs with different prompts and configurations, so they should be treated as illustrative measurements rather than a controlled benchmark.

The main idea demonstrated is that RAG can reduce the amount of document context sent to the final LLM by retrieving only relevant chunks.

**Technologies Used**
Python
OpenRouter API
OpenAI Python SDK
Sentence Transformers
all-MiniLM-L6-v2
PyTorch
Cosine Similarity
LLM-based Skill Selection
Retrieval-Augmented Generation (RAG)
Key Concepts Demonstrated
LLM-based skill routing
Skill metadata and knowledge-source mapping
Document chunking
Overlapping chunks
Text embeddings
Semantic similarity
Cosine similarity
Top-K retrieval
Context reduction
LLM-based answer generation
