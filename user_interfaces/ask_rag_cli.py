#!/usr/bin/env python3
"""
ask_rag_cli.py

A simple command-line RAG (Retrieval-Augmented Generation) assistant.

This script answers user questions using local documents and a local LLM via Ollama.
It uses semantic search to find relevant content, builds a prompt, and gets a response.

Inputs:
- User queries (typed in CLI)
- Config file: `llm_config.json` (LLM name, prompt template)
- Embedding file: `rag_vectors.json` (document chunks, embeddings, metadata)

Outputs:
- Text answers printed in the terminal
- Optional summary of training content at startup

How it works:
1. Loads LLM and embedding config
2. Loads document chunks and embeddings
3. Finds relevant content using cosine similarity
4. Builds a prompt and queries the LLM
5. Prints the result (no memory between questions)

Dependencies:
- numpy
- sentence-transformers
- scikit-learn
- ollama

Usage:
$ python3 ask_rag_cli.py
> What is this system trained on?
"""


import os
import sys
import json
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === Config Paths & Constants ===
BASE_DIR = os.path.dirname(__file__)
EMBEDDINGS_FILE = os.path.join(BASE_DIR, '..', 'pipeline', 'data', 'rag_vectors.json')
LLM_CONFIG_FILE = os.path.join(BASE_DIR, 'config', 'llm_config.json')
TOP_K = 5  # Number of top matching chunks to retrieve


def load_config():
    """
    Load LLM model name and prompt template from the config file.

    Returns:
        Tuple[str, str]: (LLM model name, prompt template string)
    """
    try:
        with open(LLM_CONFIG_FILE, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        return cfg['llm_model'], cfg['prompt_template']
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit(1)


def load_embeddings():
    """
    Load document embeddings from file and convert them to NumPy arrays.

    Returns:
        Tuple[str, List[dict]]: (embedding model name, list of chunks with embeddings)
    """
    try:
        with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for chunk in data.get('chunks', []):
            chunk['embedding'] = np.array(chunk['embedding'])
        return data.get('embed_model', 'all-MiniLM-L6-v2'), data['chunks']
    except Exception as e:
        print(f"Failed to load embeddings: {e}")
        sys.exit(1)


def retrieve(query, model, docs):
    """
    Retrieve the top-k most relevant document chunks for a given query.

    Args:
        query (str): The user's question.
        model (SentenceTransformer): The embedding model.
        docs (List[dict]): The document chunks.

    Returns:
        List[dict]: Top-k most similar document chunks.
    """
    query_vec = model.encode([query])[0].reshape(1, -1)
    doc_vecs = np.array([d['embedding'] for d in docs])
    scores = cosine_similarity(query_vec, doc_vecs)[0]
    top_ids = np.argsort(scores)[-TOP_K:][::-1]
    return [docs[i] for i in top_ids]


def make_prompt(query, chunks, template):
    """
    Fill the prompt template with retrieved context and user query.

    Args:
        query (str): User input question.
        chunks (List[dict]): Retrieved context chunks.
        template (str): Prompt template with {context} and {query} placeholders.

    Returns:
        str: Formatted prompt to send to the LLM.
    """
    context = "\n---\n".join(
        f"[{i+1}] {c['content'].strip()}\n(Source: {c.get('url', 'unknown')})"
        for i, c in enumerate(chunks)
    )
    return template.format(context=context, query=query)


def main():
    """
    Main entry point for the RAG CLI assistant.
    Loads config, embeddings, and runs interactive Q&A loop.
    """
    print("Loading RAG Assistant...\n")

    # Load model names and templates
    llm_model, prompt_template = load_config()
    embed_model_name, docs = load_embeddings()
    embed_model = SentenceTransformer(embed_model_name)

    # Display model information
    print("Model Information")
    print(f"LLM:        {llm_model}")
    print(f"Embeddings: {embed_model_name}\n")

    # On start, show summary of training content by asking LLM to list key topics
    try:
        intro_chunks = retrieve("Summarise what this system has been trained on", embed_model, docs)
        summary_prompt = (
            "List only 4 or fewer concise bullet points summarizing the main topics or document types in this content.\n"
            "Only output bullet points. Do not explain them or add headings.\n\n"
        ) + "\n---\n".join(c['content'].strip() for c in intro_chunks)

        response = ollama.chat(model=llm_model, messages=[{"role": "user", "content": summary_prompt}])
        print("Trained on:\n" + response["message"]["content"].strip() + "\n")
    except Exception as e:
        print("Couldn't summarize training data:", e)

    # User guidance
    print("Note: This assistant does not remember previous questions.")
    print("      Each response is based only on the current input.\n")

    # Interactive question loop
    print("Ask a question (type 'exit' to quit):")
    while True:
        query = input("> ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        try:
            chunks = retrieve(query, embed_model, docs)
            prompt = make_prompt(query, chunks, prompt_template)
            response = ollama.chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
            print("\n" + response["message"]["content"].strip())
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    main()
