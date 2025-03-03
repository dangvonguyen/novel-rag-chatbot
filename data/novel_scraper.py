import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests  # type: ignore
from bs4 import BeautifulSoup, Tag


@dataclass
class NovelMetadata:
    """Data class to store novel metadata."""

    title: str
    author: str
    genre: list[str]
    description: str

    def to_dict(self) -> dict[str, str | list[str]]:
        """Convert metadata to dictionary format."""
        return {
            "Tên truyện": self.title,
            "Tác giả": self.author,
            "Thể loại": self.genre,
            "Giới thiệu": self.description,
        }


class NovelScraper:
    def __init__(
        self, novel_url: str, save_dir: str = "data", delay: float = 1.0
    ) -> None:
        assert novel_url, "Novel URL cannot be empty"
        assert delay >= 0, "Delay must be non-negative"

        # Set up logging
        self._setup_logging()

        self.novel_url = novel_url
        self.save_dir = Path(save_dir)
        self.delay = delay

        # Initialize directory structure
        self._setup_directories()

        # Save novel metadata
        self.save_metadata()

    def _setup_logging(self) -> None:
        """Configure logging for the scraper."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def _setup_directories(self) -> None:
        """Create necessary directories for saving content."""
        try:
            self.novel_dir = self.save_dir / self.novel_url.strip("/").split("/")[-1]
            self.chapter_dir = self.novel_dir / "chapters"

            self.novel_dir.mkdir(parents=True, exist_ok=True)
            self.chapter_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Failed to create directories: {str(e)}")

    def get_page_content(self, url: str) -> str | None:
        """Fetch the HTML content of a webpage."""
        try:
            time.sleep(self.delay)  # Rate limiting
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return str(response.text)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def _parse_metadata(self, soup: BeautifulSoup) -> NovelMetadata:
        """Parse novel metadata from BeautifulSoup object."""
        try:
            title_elem = soup.find(attrs={"class": "title"})
            if not isinstance(title_elem, Tag):
                raise ValueError("Title element not found")
            title = title_elem.text.strip()

            info_div = soup.find("div", {"class": "info"})
            if not isinstance(info_div, Tag):
                raise ValueError("Info div not found")

            # Parse author and genre from info
            author_elem = info_div.find(attrs={"itemprop": "author"})
            if not isinstance(author_elem, Tag):
                raise ValueError("Author element not found")
            author = author_elem.text.strip()

            genre_tags = info_div.find_all(attrs={"itemprop": "genre"})
            genre = [tag.text.strip() for tag in genre_tags if isinstance(tag, Tag)]

            desc_elem = soup.find(attrs={"class": "desc-text"})
            if not isinstance(desc_elem, Tag):
                raise ValueError("Description element not found")
            description = desc_elem.text.strip()

            return NovelMetadata(title, author, genre, description)
        except (AttributeError, TypeError) as e:
            raise ValueError(f"Failed to parse metadata: {str(e)}")

    def save_metadata(self) -> None:
        """Fetch and save novel metadata to a JSON file."""
        html = self.get_page_content(self.novel_url)
        if html is None:
            raise ValueError(f"Failed to fetch content from {self.novel_url}")

        soup = BeautifulSoup(html, "html.parser")
        metadata = self._parse_metadata(soup)

        try:
            with open(self.novel_dir / "metadata.json", "w", encoding="utf-8") as fp:
                json.dump(metadata.to_dict(), fp, ensure_ascii=False, indent=4)
        except OSError as e:
            raise ValueError(f"Failed to save metadata: {str(e)}")

    def _parse_content(self, soup: BeautifulSoup) -> str | None:
        """Parse content from HTML based on CSS selectors."""
        try:
            content_elem = soup.find(attrs={"class": "chapter-c"})
            if not isinstance(content_elem, Tag):
                return None
            return content_elem.get_text(" ", strip=True)
        except AttributeError:
            return None

    def scrape_chapter(self, chapter: int, is_save: bool = True) -> str | None:
        """Scrape content from a specific chapter and optionally save to a file."""
        chapter_url = self.novel_url + f"chuong-{chapter}"
        html = self.get_page_content(chapter_url)
        if html is None:
            return None

        soup = BeautifulSoup(html, "html.parser")
        content = self._parse_content(soup)
        if content is None:
            return None

        if is_save:
            chapter_path = self.chapter_dir / f"{chapter}.txt"
            with open(chapter_path, "w", encoding="utf-8") as fp:
                fp.write(content)

        return content

    def scrape_multiple_chapters(
        self, start: int, end: int, is_save: bool = True
    ) -> list[str | None]:
        """Scrape multiple chapters and optionally save to files."""
        return [
            self.scrape_chapter(chapter, is_save) for chapter in range(start, end + 1)
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape novel content from a website.")
    parser.add_argument(
        "-u", "--novel-url", type=str, help="URL of the novel to scrape."
    )
    parser.add_argument(
        "--save-dir", type=str, default="data", help="Directory to save content."
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Delay between requests (in seconds)."
    )
    parser.add_argument(
        "--start", type=int, default=1, help="Starting chapter to scrape."
    )
    parser.add_argument("--end", type=int, default=1, help="Ending chapter to scrape.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scraper = NovelScraper(args.novel_url, args.save_dir, args.delay)
    scraper.scrape_multiple_chapters(start=args.start, end=args.end, is_save=True)
