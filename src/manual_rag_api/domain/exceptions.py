"""Domain exceptions for manual_rag_api."""

class IndexNotFoundError(RuntimeError):
    """Raised when the LanceDB index does not exist."""

class PDFNotFoundError(FileNotFoundError):
    """Raised when the source PDF cannot be found."""

class SearchError(RuntimeError):
    """Raised when retrieval fails unrecoverably."""

class GenerationError(RuntimeError):
    """Raised when LLM answer generation fails."""
