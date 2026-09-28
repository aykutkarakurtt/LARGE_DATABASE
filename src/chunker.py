import re


def chunk_text(text, chunk_size=500, overlap=50):
    if chunk_size <= 0:
        raise ValueError("chunk_size pozitif olmalı")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap, chunk_size'dan küçük ve negatif olmamalı")

    text = _normalize_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            boundary = _find_break(text, start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(start + 1, end - overlap)
        next_start = _move_to_word_start(text, next_start)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def _normalize_text(text):
    text = text.replace("\xad", "")
    return re.sub(r"\s+", " ", text).strip()


def _find_break(text, start, end):
    lower_bound = start + max(1, (end - start) // 2)
    candidates = [
        text.rfind(" ", lower_bound, end),
        text.rfind(".", lower_bound, end),
        text.rfind("!", lower_bound, end),
        text.rfind("?", lower_bound, end),
    ]
    candidates = [
        candidate + 1
        for candidate in candidates
        if candidate >= lower_bound
    ]
    return max(candidates) if candidates else end


def _move_to_word_start(text, position):
    if position <= 0 or position >= len(text):
        return position
    if text[position].isspace() or text[position - 1].isspace():
        return position
    space = text.rfind(" ", 0, position)
    return space + 1 if space >= 0 else position


def create_chunks(document, chunk_size=500, chunk_overlap=50):
    chunks = []
    source = document["source"]
    for page_data in document["pages"]:
        page_chunks = chunk_text(
            page_data["text"],
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )
        for chunk_index, chunk in enumerate(page_chunks):
            chunks.append({
                "text": chunk,
                "source": source,
                "page": page_data["page"],
                "chunk_index": chunk_index,
            })
    return chunks
