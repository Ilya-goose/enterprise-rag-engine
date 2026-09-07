def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    if not text: return []
    chunk_list = list()
    start = 0
    end = chunk_size
    while end <= len(text):
        right_space = text.rfind(' ', start, end)
        if right_space == -1:
            right_space = end
        chunk_list.append(text[start:right_space])
        overlap_start = max(right_space - overlap, start)
        overlap_space = text.find(' ', overlap_start, right_space)
        if overlap_space != -1:
            start = overlap_space + 1
        else:
            start = max(right_space - overlap, start + 1)
        end = start + chunk_size
    chunk_list.append(text[start:])
    return chunk_list