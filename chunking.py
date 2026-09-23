"""
Text chunking, kept free of model/SDK imports so the evaluation harness
(eval/) can reuse the exact same chunker the app uses at ingestion time.
"""

import config


def chunk_text(text: str, max_chars: int = None):
    """
    Split text into chunks along paragraph breaks, merging short paragraphs
    together and splitting any paragraph that's still too long.
    """
    max_chars = max_chars or config.CHUNK_MAX_CHARS
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(para) <= max_chars:
                current = para
            else:
                # Hard-split an overly long paragraph
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i:i + max_chars])
                current = ""
    if current:
        chunks.append(current)
    return chunks
