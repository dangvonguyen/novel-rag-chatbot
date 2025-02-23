# Novel RAG Chatbot

## Overview

The **Novel RAG Chatbot** is a conversational AI chatbot that leverages Retrieval-Augmented Generation (RAG) for dynamic, context-driven dialogue and storytelling. It is designed to assist users with inquiries related to novels, providing precise and context-aware responses.

## Features

- **Dynamic Conversations**: Engage in meaningful dialogues about novels, characters, and plots.
- **Contextual Understanding**: The chatbot can classify user inquiries and respond appropriately based on the context.
- **Document Retrieval**: It can retrieve relevant information from a collection of documents to enhance responses.
- **Customizable**: Easily configure the chatbot's behavior and responses through a configuration file.

## Requirements

- [uv](https://github.com/astral-sh/uv) package manager (recommend)

## Installation

1. Clone the repository:

```sh
$ git clone https://github.com/yourusername/novel-rag-chatbot.git
$ cd novel-rag-chatbot
```

2. Create a virtual environment and install the required dependencies:

```sh
$ uv sync
```

3. Set up your environment variables and fill in your API keys:

```sh
$ cp .env.example .env
```

## Usage

1. Crawl your novels and save them to the data directory:

```sh
$ python data/novel_scraper.py --url NOVEL_URL --start START_CHAPTER --end END_CHAPTER
```

2. Add documents to the vector store:

```sh
$ python add_docs.py
```

3. Run the application:

```sh
$ streamlit run app.py
```

This will start a local server, and you can access the chatbot through your web browser.

## Configuration

The chatbot's behavior can be customized through the `src/configs/configuration.py` file. You can adjust parameters such as:

- `embedding_model`: The model used for embeddings.
- `splitter_type`: The strategy for splitting documents.
- `chunk_size`: The size of each document chunk.
- `response_model`: The language model used for generating responses.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.
