import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.configs import Configuration
from src.utils import get_literal_values


class DocumentManager:

    def __init__(self):
        self.splitter = self._make_splitter()

    def _load_metadata(self, comic_dir: Path):
        fpath = comic_dir / "metadata.json"
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_chapter_content(self, chapter_path: Path):
        with open(chapter_path, "r", encoding="utf-8") as f:
            return f.read()

    def load_from_dir(self, comic_dir: Path) -> List[Document]:
        docs = []

        # Load metadata
        metadata = self._load_metadata(comic_dir)

        # Load chapters
        chapter_dir = comic_dir / "chapters"
        for fpath in sorted(chapter_dir.iterdir(), key=lambda x: int(x.stem)):
            content = self._load_chapter_content(fpath)
            doc = Document(
                page_content=content, metadata={**metadata, "Chương": fpath.stem}
            )
            docs.append(doc)

        return docs

    def load_documents(self, data_dir: str) -> List[Document]:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {data_dir}")

        docs = []
        for comic_dir in data_dir.iterdir():
            if comic_dir.is_dir():
                docs.extend(self.load_from_dir(comic_dir))
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        match self.splitter_type:
            case "delimiter":
                return self.splitter.split_documents(documents)

            case "semantic":
                all_chunked_docs = []
                for doc in documents:
                    metadata = doc.metadata
                    content = doc.page_content
                    chunked_content = self.splitter.chunks(content)
                    chunked_docs = [
                        Document(chunk, metadata=metadata) for chunk in chunked_content
                    ]
                    all_chunked_docs.extend(chunked_docs)
                return all_chunked_docs

    def _make_splitter(self):
        configuration = Configuration.from_runnable_config()
        self.splitter_type = configuration.splitter_type
        match self.splitter_type:
            case "delimiter":
                from langchain_text_splitters import RecursiveCharacterTextSplitter

                return RecursiveCharacterTextSplitter(
                    chunk_size=configuration.chunk_size,
                    chunk_overlap=configuration.chunk_overlap,
                )

            case "semantic":
                from semantic_text_splitter import TextSplitter
                from tokenizers import Tokenizer

                tokenizer = Tokenizer.from_pretrained(configuration.semantic_tokenizer)
                return TextSplitter.from_huggingface_tokenizer(
                    tokenizer=tokenizer,
                    capacity=configuration.chunk_size,
                    overlap=configuration.chunk_overlap,
                )

            case _:
                raise ValueError(
                    "Unrecognized vectorstore_provider in configuration. "
                    f"Expected one of: {get_literal_values(configuration, 'retriever_provider')}\n"
                    f"Got: {configuration.splitter_type}"
                )
