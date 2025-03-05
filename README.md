# Vectorstore

## Overview

This project demonstrates how to use Langchain's document loaders, text splitters, and vector stores to manage and query text data efficiently. It utilizes OpenAI embeddings for vectorization and supports both FAISS and Chroma vector stores for similarity search operations. The primary goal is to showcase how to load text data, process it into manageable chunks, store it as vectors, and perform similarity searches based on user queries.

## Features

- **Document Loading**: Utilizes TextLoader to read text files and prepare them for processing.
- **Text Splitting**: Implements RecursiveCharacterTextSplitter to divide large documents into smaller, manageable chunks with defined overlap.
- **Vectorization**: Uses OpenAIEmbeddings to convert text chunks into vector representations suitable for similarity searches.
- **Vector Store Management**:
  - **FAISS**: Demonstrates how to create a FAISS vector store from documents, perform similarity searches, and save/load the vector store locally.
  - **Chroma**: Illustrates the creation of a Chroma vector store, performing similarity searches, and saving/loading from disk.
- **Similarity Search**: Provides functionality to search for similar documents based on a given query string or vector representation.
