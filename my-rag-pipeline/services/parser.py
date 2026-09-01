import io
import logging
from typing import Union

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_bytes_or_path: Union[bytes, str, io.BytesIO]) -> str:
    """Extract text from a PDF provided as bytes or a filesystem path.

    Returns the concatenated text from all pages. Errors during page extraction
    are caught and logged; the function will continue processing remaining pages.
    In case of an unrecoverable error, an empty string is returned.
    """
    try:
        # Accept raw bytes or a file path/like object
        if isinstance(file_bytes_or_path, (bytes, bytearray)):
            stream = io.BytesIO(file_bytes_or_path)
            reader = PdfReader(stream)
        else:
            reader = PdfReader(file_bytes_or_path)

        pages_text = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                logger.exception("Error extracting text from page %s: %s", i, e)
                text = ""
            pages_text.append(text)

        return "\n".join(pages_text)

    except Exception as exc:
        logger.exception("Failed to read PDF: %s", exc)
        return ""
