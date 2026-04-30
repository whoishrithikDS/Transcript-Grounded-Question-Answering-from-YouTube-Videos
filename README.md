Transcript-Grounded Question Answering from YouTube Videos
📌 Overview

AskTube AI is an LLM-powered question-answering system that allows users to ask natural language questions about a YouTube video and receive answers strictly grounded in the video’s transcript.

The system:

Fetches YouTube captions

Splits and embeds the transcript

Stores embeddings in a vector database

Retrieves only relevant context

Forces the LLM to answer only from the transcript

Returns “I don’t know” if the answer is not present

This prevents hallucination and keeps answers verifiable.

🧠 Why this project matters

Most LLM demos answer anything — even when the source doesn’t contain the answer.

This project:

Enforces source-grounded answering

Demonstrates RAG (Retrieval-Augmented Generation)

Uses local LLMs (Ollama) instead of paid APIs

Is directly applicable to education, research, and interviews

🏗️ Architecture
YouTube Video
     ↓
Transcript Extraction
     ↓
Text Chunking
     ↓
Embedding Generation (Ollama)
     ↓
FAISS Vector Store
     ↓
Similarity Retrieval
     ↓
Context-Bound Prompt
     ↓
LLM Answer (No Hallucination)

🛠️ Tech Stack

Python

LangChain

FAISS (vector database)

Ollama

qwen3:4b (LLM)

qwen3-embedding:0.6b (embeddings)

YouTube Transcript API

✨ Key Features

🔍 Ask questions about any YouTube video with captions

🧠 Answers only from retrieved transcript chunks

🚫 Hallucination control via strict prompting

⚡ Fast local inference using Ollama

📄 Modular, readable code

🚀 How it Works (Step-by-Step)

Fetch transcript using video ID

Split transcript into overlapping chunks

Generate embeddings using Ollama

Store embeddings in FAISS

Retrieve top-k relevant chunks for a question

Inject retrieved context into a constrained prompt

Generate answer using local LLM..
