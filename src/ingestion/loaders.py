"""Document loaders for the ingestion pipeline.

Wraps ``langchain_community`` document loaders behind a single async
interface. LangChain's file and web loaders are synchronous, so they are
dispatched to a worker thread via :func:`asyncio.to_thread` to keep the
event loop free for other coroutines.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)

from src.config.constants import SourceType
from src.ingestion import IngestionError, UnsupportedSourceError

if TYPE_CHECKING:
    from langchain_core.document_loaders import BaseLoader
    from langchain_core.documents import Document

logger = structlog.get_logger(__name__)

_EXTENSION_TO_SOURCE_TYPE: dict[str, SourceType] = {
    ".pdf": SourceType.PDF,
    ".txt": SourceType.TEXT,
    ".text": SourceType.TEXT,
    ".csv": SourceType.CSV,
    ".md": SourceType.MARKDOWN,
    ".markdown": SourceType.MARKDOWN,
}


def detect_source_type(source: str) -> SourceType:
    """Infer the :class:`SourceType` from a file path or URL.

    Args:
        source: Local file path or HTTP(S) URL.

    Returns:
        The matching source type enum value.

    Raises:
        UnsupportedSourceError: If the URL scheme is unsupported or the
            file extension is unknown.
    """
    if source.lower().startswith(("http://", "https://")):
        return SourceType.WEB
    suffix = Path(source).suffix.lower()
    try:
        return _EXTENSION_TO_SOURCE_TYPE[suffix]
    except KeyError as exc:
        raise UnsupportedSourceError(
            f"Cannot determine loader for {source!r} (extension {suffix!r})"
        ) from exc


class DocumentLoader:
    """Load documents from PDF, text, CSV, markdown, and web sources."""

    async def load(self, source: str, source_type: SourceType | None = None) -> list[Document]:
        """Load a single source, dispatching on its detected type.

        Args:
            source: Local file path or HTTP(S) URL.
            source_type: Override auto-detected source type.

        Returns:
            Loaded LangChain documents with non-empty content.

        Raises:
            UnsupportedSourceError: If no loader matches the source.
            IngestionError: If the underlying loader fails.
        """
        if source_type is None:
            source_type = detect_source_type(source)
        if source_type is SourceType.PDF:
            return await self.load_pdf(source)
        if source_type is SourceType.TEXT:
            return await self.load_text(source)
        if source_type is SourceType.CSV:
            return await self.load_csv(source)
        if source_type is SourceType.MARKDOWN:
            return await self.load_markdown(source)
        if source_type is SourceType.WEB:
            return await self.load_web(source)
        raise UnsupportedSourceError(f"No loader registered for {source_type!r}")

    async def load_pdf(self, source: str) -> list[Document]:
        """Load a PDF, producing one document per page.

        Requires the optional ``pypdf`` dependency at load time.
        """
        return await self._run_loader(PyPDFLoader(source), source)

    async def load_text(self, source: str) -> list[Document]:
        """Load a plain-text file as a single document."""
        loader = TextLoader(source, encoding="utf-8", autodetect_encoding=True)
        return await self._run_loader(loader, source)

    async def load_csv(self, source: str) -> list[Document]:
        """Load a CSV file, producing one document per row."""
        loader = CSVLoader(source, encoding="utf-8")
        return await self._run_loader(loader, source)

    async def load_markdown(self, source: str) -> list[Document]:
        """Load a markdown file.

        Prefers the unstructured markdown loader and falls back to plain
        text loading when the optional ``unstructured`` dependency is
        missing, so markdown ingestion still works in minimal installs.
        """
        try:
            loader: BaseLoader = UnstructuredMarkdownLoader(source)
        except ModuleNotFoundError:
            logger.warning(
                "unstructured unavailable; falling back to TextLoader",
                source=source,
            )
            loader = TextLoader(source, encoding="utf-8", autodetect_encoding=True)
        return await self._run_loader(loader, source)

    async def load_web(self, source: str) -> list[Document]:
        """Load and parse a web page from an HTTP(S) URL."""
        return await self._run_loader(WebBaseLoader(source), source)

    async def _run_loader(self, loader: BaseLoader, source: str) -> list[Document]:
        """Execute a synchronous LangChain loader in a worker thread."""
        try:
            documents = await asyncio.to_thread(loader.load)
        except IngestionError:
            raise
        except Exception as exc:  # noqa: BLE001 - backend error types are not stable
            raise IngestionError(f"Failed to load {source!r}: {exc}") from exc
        documents = [doc for doc in documents if doc.page_content.strip()]
        logger.info("Document loaded", source=source, document_count=len(documents))
        return documents
