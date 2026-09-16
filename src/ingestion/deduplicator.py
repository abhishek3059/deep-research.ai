"""Content-hash based document deduplication.

Deduplication happens before embedding so identical content is never sent
to the embedding provider twice.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = structlog.get_logger(__name__)


class Deduplicator:
    """Remove duplicate documents by the SHA-256 hash of their content."""

    @staticmethod
    def compute_hash(text: str) -> str:
        """Return the hex-encoded SHA-256 digest of ``text``.

        Args:
            text: Raw text content to hash.

        Returns:
            64-character lowercase hexadecimal digest.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def deduplicate(self, documents: list[Document]) -> list[Document]:
        """Return documents with duplicate content removed.

        The first occurrence of each unique content hash is kept, so the
        input order is preserved.

        Args:
            documents: Documents to deduplicate.

        Returns:
            Unique documents in first-seen order.
        """
        seen: set[str] = set()
        unique: list[Document] = []
        for document in documents:
            digest = self.compute_hash(document.page_content)
            if digest in seen:
                continue
            seen.add(digest)
            unique.append(document)
        logger.info(
            "Documents deduplicated",
            input_documents=len(documents),
            unique_documents=len(unique),
            removed=len(documents) - len(unique),
        )
        return unique

    def filter_new(
        self,
        documents: list[Document],
        existing_hashes: set[str],
    ) -> list[Document]:
        """Return only documents whose content hash is not in ``existing_hashes``.

        Args:
            documents: Documents to filter.
            existing_hashes: Set of SHA-256 hex digests already ingested.

        Returns:
            Documents with new (unseen) content, in input order.
        """
        new_docs: list[Document] = []
        for document in documents:
            digest = self.compute_hash(document.page_content)
            if digest not in existing_hashes:
                new_docs.append(document)
        logger.info(
            "New documents filtered",
            input_documents=len(documents),
            new_documents=len(new_docs),
            skipped=len(documents) - len(new_docs),
        )
        return new_docs
