import json
import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentManager:

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

    def _load_metadata(self, comic_dir: Path):
        fpath = comic_dir / "metadata.json"
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_chapter_content(self, chapter_path: Path):
        with open(chapter_path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_from_dir(self, comic_dir: Path) -> List[Document]:
        docs = []

        # Load metadata
        metadata = self._load_metadata(comic_dir)

        # Load chapters
        chapter_dir = comic_dir / 'chapters'
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
        return self.splitter.split_documents(documents)