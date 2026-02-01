"""
Document Processor

Reads and normalizes documents from various formats:
- PDF files (AAD guidelines)
- JSON files (structured StatPearls data)
- HTML/Text files (StatPearls web content)
- Markdown files

Extracts metadata (title, source, section) for each document.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import re

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class Document:
    """Represents a processed document with text and metadata."""

    def __init__(
        self,
        text: str,
        metadata: Dict[str, str],
        doc_id: Optional[str] = None
    ):
        self.text = text
        self.metadata = metadata
        self.doc_id = doc_id or self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique ID from metadata."""
        source = self.metadata.get("source", "unknown")
        title = self.metadata.get("title", "untitled")
        section = self.metadata.get("section", "")

        # Create simple slug
        slug = f"{source}_{title}_{section}".lower()
        slug = re.sub(r'[^a-z0-9_]+', '_', slug)
        return slug[:100]  # Limit length

    def __repr__(self):
        return f"Document(id={self.doc_id}, source={self.metadata.get('source')}, title={self.metadata.get('title')[:50]}...)"


class DocumentProcessor:
    """Process documents from multiple formats."""

    def __init__(self):
        """Initialize document processor."""
        if PdfReader is None:
            logger.warning("PyPDF2 not installed. PDF reading will be disabled.")

    def process_file(self, file_path: Path) -> List[Document]:
        """
        Process a single file based on its extension.

        Args:
            file_path: Path to file

        Returns:
            List of Document objects
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []

        suffix = file_path.suffix.lower()

        try:
            if suffix == '.pdf':
                return self.process_pdf(file_path)
            elif suffix == '.json':
                return self.process_json(file_path)
            elif suffix in ['.html', '.htm']:
                return self.process_html(file_path)
            elif suffix in ['.txt', '.md']:
                return self.process_text(file_path)
            else:
                logger.warning(f"Unsupported file type: {suffix}")
                return []
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []

    def process_pdf(self, file_path: Path) -> List[Document]:
        """
        Extract text from PDF file.

        Args:
            file_path: Path to PDF

        Returns:
            List of Document objects (one per page or section)
        """
        if PdfReader is None:
            logger.error("PyPDF2 not installed. Cannot read PDF.")
            return []

        try:
            reader = PdfReader(str(file_path))
            documents = []

            # Extract metadata from filename
            base_metadata = {
                "source": "AAD",
                "title": file_path.stem,
                "file_type": "pdf",
                "file_path": str(file_path)
            }

            # Process each page
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()

                if text.strip():  # Only add non-empty pages
                    metadata = base_metadata.copy()
                    metadata["page"] = str(page_num)
                    metadata["section"] = f"Page {page_num}"

                    documents.append(Document(
                        text=text.strip(),
                        metadata=metadata
                    ))

            logger.info(f"Extracted {len(documents)} pages from {file_path.name}")
            return documents

        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return []

    def _extract_text_from_nested(self, obj, max_depth=10, current_depth=0):
        """
        Recursively extract text from nested JSON structures.

        Args:
            obj: Any JSON object (dict, list, str, etc.)
            max_depth: Maximum recursion depth
            current_depth: Current depth (internal)

        Returns:
            Extracted text as string
        """
        if current_depth > max_depth:
            return ""

        if isinstance(obj, str):
            return obj
        elif isinstance(obj, (int, float, bool)):
            return str(obj)
        elif isinstance(obj, list):
            # Join list items with newlines
            texts = [self._extract_text_from_nested(item, max_depth, current_depth + 1)
                     for item in obj]
            return "\n".join(text for text in texts if text)
        elif isinstance(obj, dict):
            # Extract text from all dict values
            texts = []
            for key, value in obj.items():
                # Skip metadata keys
                if key in ["source", "document_type", "year", "file_type", "file_path"]:
                    continue
                text = self._extract_text_from_nested(value, max_depth, current_depth + 1)
                if text:
                    texts.append(text)
            return "\n\n".join(texts)
        else:
            return ""

    def process_json(self, file_path: Path) -> List[Document]:
        """
        Extract text from JSON file with support for multiple formats.

        Handles:
        - AAD format: nested dicts with "content" key
        - StatPearls format: "sections" dict with arrays
        - Generic JSON: any nested structure

        Args:
            file_path: Path to JSON

        Returns:
            List of Document objects
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            documents = []

            # Determine source from file path
            source = "StatPearls" if "StatPearls" in str(file_path) else "AAD"

            # Extract title from various possible locations
            title = None
            if isinstance(data.get("title"), str):
                title = data["title"]
            elif isinstance(data.get("source"), dict) and "guideline_title" in data["source"]:
                title = data["source"]["guideline_title"]
            elif "condition" in data:
                title = data["condition"]
            else:
                title = file_path.stem

            # Strategy 1: StatPearls format - sections dict with arrays
            if "sections" in data and isinstance(data["sections"], dict):
                for section_name, section_content in data["sections"].items():
                    if isinstance(section_content, list):
                        # Join array items
                        text = "\n\n".join(str(item) for item in section_content if item)
                    else:
                        text = self._extract_text_from_nested(section_content)

                    if text.strip():
                        metadata = {
                            "source": source,
                            "title": title,
                            "section": section_name,
                            "file_type": "json",
                            "file_path": str(file_path)
                        }
                        documents.append(Document(text=text.strip(), metadata=metadata))

            # Strategy 2: AAD format - content dict
            elif "content" in data:
                content_obj = data["content"]
                text = self._extract_text_from_nested(content_obj)

                if text.strip():
                    section_name = data.get("section", "Main Content")
                    metadata = {
                        "source": source,
                        "title": title,
                        "section": section_name,
                        "file_type": "json",
                        "file_path": str(file_path)
                    }
                    documents.append(Document(text=text.strip(), metadata=metadata))

            # Strategy 3: Generic nested structure
            else:
                text = self._extract_text_from_nested(data)

                if text.strip():
                    metadata = {
                        "source": source,
                        "title": title,
                        "file_type": "json",
                        "file_path": str(file_path)
                    }
                    documents.append(Document(text=text.strip(), metadata=metadata))

            logger.info(f"Extracted {len(documents)} sections from {file_path.name}")
            return documents

        except Exception as e:
            logger.error(f"Error reading JSON {file_path}: {e}")
            return []

    def process_html(self, file_path: Path) -> List[Document]:
        """
        Extract text from HTML file.

        Args:
            file_path: Path to HTML

        Returns:
            List of Document objects
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text() if title_tag else file_path.stem

            # Get main text
            text = soup.get_text(separator='\n', strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = '\n'.join(lines)

            if text:
                metadata = {
                    "source": "StatPearls",
                    "title": title,
                    "file_type": "html",
                    "file_path": str(file_path)
                }
                return [Document(text=text, metadata=metadata)]

            return []

        except Exception as e:
            logger.error(f"Error reading HTML {file_path}: {e}")
            return []

    def process_text(self, file_path: Path) -> List[Document]:
        """
        Extract text from plain text or markdown file.

        Args:
            file_path: Path to text file

        Returns:
            List of Document objects
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if text.strip():
                metadata = {
                    "source": "Unknown",
                    "title": file_path.stem,
                    "file_type": file_path.suffix[1:],  # Remove dot
                    "file_path": str(file_path)
                }
                return [Document(text=text.strip(), metadata=metadata)]

            return []

        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return []

    def process_directory(
        self,
        directory: Path,
        recursive: bool = True,
        file_pattern: str = "*"
    ) -> List[Document]:
        """
        Process all files in a directory.

        Args:
            directory: Path to directory
            recursive: Whether to process subdirectories
            file_pattern: Glob pattern for files (e.g., "*.pdf")

        Returns:
            List of Document objects
        """
        directory = Path(directory)
        documents = []

        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return []

        # Get files
        if recursive:
            files = directory.rglob(file_pattern)
        else:
            files = directory.glob(file_pattern)

        # Process each file
        for file_path in files:
            if file_path.is_file():
                docs = self.process_file(file_path)
                documents.extend(docs)

        logger.info(f"Processed {len(documents)} documents from {directory}")
        return documents
