"""
Persistent document registry for incremental indexing.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class DocumentRegistry:
    """Keeps track of indexed documents."""

    def __init__(
        self,
        registry_path: str = ".knowledge/index/manifest.json",
    ) -> None:

        self.registry_path = Path(registry_path)
        self.documents: Dict[str, Dict[str, Any]] = {}

        self.load()

    # -------------------------------------------------------
    # Internal helper
    # -------------------------------------------------------

    @staticmethod
    def _key(path: Path) -> str:
        """
        Return a canonical key for every file.

        Using resolved absolute paths avoids duplicate entries caused by
        relative vs absolute paths.
        """
        return str(path.resolve())

    # -------------------------------------------------------
    # Persistence
    # -------------------------------------------------------

    def load(self) -> None:
        """Load registry."""

        if not self.registry_path.exists():
            self.documents = {}
            return

        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.documents = data.get("documents", {})

    def save(self) -> None:
        """Save registry."""

        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "documents": self.documents,
                },
                f,
                indent=4,
            )

    # -------------------------------------------------------
    # Hashing
    # -------------------------------------------------------

    @staticmethod
    def compute_hash(path: Path) -> str:
        """Compute SHA256 hash."""

        sha = hashlib.sha256()

        with open(path, "rb") as f:

            while True:

                block = f.read(8192)

                if not block:
                    break

                sha.update(block)

        return sha.hexdigest()

    # -------------------------------------------------------
    # CRUD
    # -------------------------------------------------------

    def add_document(
        self,
        path: Path,
        chunk_count: int,
    ) -> None:

        key = self._key(path)

        self.documents[key] = {
            "hash": self.compute_hash(path),
            "chunks": chunk_count,
            "modified": path.stat().st_mtime,
        }

    def remove_document(
        self,
        path: Path,
    ) -> None:

        key = self._key(path)

        self.documents.pop(key, None)

    def get_document(
        self,
        path: Path,
    ) -> Optional[Dict[str, Any]]:

        key = self._key(path)

        return self.documents.get(key)

    # -------------------------------------------------------
    # Status
    # -------------------------------------------------------

    def get_status(
        self,
        path: Path,
    ) -> str:
        """
        Returns

        NEW
        MODIFIED
        UNCHANGED
        """

        record = self.get_document(path)

        if record is None:
            return "NEW"

        current_hash = self.compute_hash(path)

        if current_hash != record["hash"]:
            return "MODIFIED"

        return "UNCHANGED"


    def deleted_documents(
        self,
        current_files: List[Path],
    ) -> List[str]:

        current = {
            self._key(path)
            for path in current_files
        }

        deleted = []

        for file in self.documents:

            if file not in current:
                deleted.append(file)

        return deleted