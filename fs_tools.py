import os
import re
from pathlib import Path
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]


class FileListInput(BaseModel):
    directory: str = Field(description="Directory to list files from.")
    extension: Optional[str] = Field(
        default=None,
        description="Optional extension filter, e.g. '.pdf' or 'pdf'. "
                    "Case-insensitive. If None, all files are listed."
    )
def list_files(directory: str, extension: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List files in a directory, optionally filtered by extension.

    Args:
        directory: Directory to list files from.
        extension: Optional extension filter, e.g. ".pdf" or "pdf".
                   Case-insensitive. If None, all files are listed.

    Returns:
        A list of dicts, one per file:
        {
            "name": str,
            "filepath": str,
            "extension": str,
            "size_bytes": int,
            "modified": str,   # ISO 8601 timestamp
        }
        Returns [{"error": "..."}] if the directory can't be read.
    """
    if not os.path.isdir(directory):
        return [{"error": f"Directory not found: {directory}"}]

    if extension is not None and not extension.startswith("."):
        extension = "." + extension
    if extension is not None:
        extension = extension.lower()

    files_info: List[Dict[str, Any]] = []

    try:
        for entry in sorted(os.listdir(directory)):
            full_path = os.path.join(directory, entry)
            if not os.path.isfile(full_path):
                continue

            ext = _get_extension(entry)
            if extension is not None and ext != extension:
                continue

            try:
                stat = os.stat(full_path)
                files_info.append({
                    "name": entry,
                    "filepath": full_path,
                    "extension": ext,
                    "size_bytes": stat.st_size
                })
            except OSError:
                continue  # skip files we can't stat (e.g. broken symlinks)

    except PermissionError:
        return [{"error": f"Permission denied: {directory}"}]
    except Exception as e:
        return [{"error": f"Failed to list directory: {e}"}]

    return files_info

class FileReadInput(BaseModel):
    filepath: str = Field(description="Path to the file to read.")

def read_file(filepath: str) -> Dict[str, Any]:
    """
    Read a resume file and extract its text.

    Supports PDF, TXT and DOCX files.

    Args:
        filepath: Path to the file that should be read.

    Returns:
        A dictionary containing file metadata and extracted text.
    """

    try:
        path = Path(filepath)

        if not path.exists():
            return {
                "success": False,
                "error": f"File does not exist: {filepath}"
            }

        if not path.is_file():
            return {
                "success": False,
                "error": f"Path is not a file: {filepath}"
            }

        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            return {
                "success": False,
                "error": (
                    f"Unsupported file type: {extension}. "
                    f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
                )
            }

        content = _get_loader(filepath, extension)

        stat = path.stat()

        return {
            "success": True,
            "filepath": str(path),
            "filename": path.name,
            "extension": extension,
            "size_bytes": stat.st_size,
            "modified_date": datetime.fromtimestamp(
                stat.st_mtime
            ).isoformat(),
            "content": content.load(),
            "character_count": len(content.load()),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to read file: {str(e)}"
        }



def _get_loader(filepath: str, ext: str):
    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(filepath)

    if ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(filepath)

    if ext in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader
        return TextLoader(filepath, autodetect_encoding=True)


def _get_extension(filepath: str) -> str:
    return os.path.splitext(filepath)[1].lower()
