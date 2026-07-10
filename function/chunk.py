"""文本分块工具 —— 读取纯文本并按指定大小分块，支持重叠窗口、节边界感知、进度回调。"""

from pathlib import Path
from typing import Callable
import re as _re


def load_and_chunk(
    file_path: str,
    max_chunk_size: int = 2000,
    chunk_overlap: int = 0,
    min_chunk_size: int = 0,
    min_paragraph_size: int = 50,
    preserve_sections: bool = True,
    encoding_fallbacks: list[str] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[str]:
    """读取文本文件并按指定大小分块。

    流程: 读取(UTF-8→GBK→Latin-1) → 规范化 → 合并短段落 → 按max_chunk_size组合。
    支持重叠窗口(chunk_overlap)、节边界感知(preserve_sections)、进度回调。
    """
    if max_chunk_size < 1:
        raise ValueError(f"max_chunk_size 必须 >= 1，当前值: {max_chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap 必须 >= 0，当前值: {chunk_overlap}")
    if chunk_overlap >= max_chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) 必须小于 max_chunk_size ({max_chunk_size})"
        )
    if encoding_fallbacks is None:
        encoding_fallbacks = ["gbk", "latin-1"]

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    if not path.is_file():
        raise ValueError(f"路径不是文件: {file_path}")

    text = _read_with_fallback(path, encoding_fallbacks)
    _report_progress(on_progress, 1, 4)

    text = _normalize_whitespace(text)
    raw_paragraphs = text.split("\n\n")
    _report_progress(on_progress, 2, 4)

    seps = _build_section_pattern() if preserve_sections else None
    paragraphs = _merge_short_paragraphs(raw_paragraphs, min_paragraph_size, seps)
    _report_progress(on_progress, 3, 4)

    chunks = _build_chunks(paragraphs, max_chunk_size, chunk_overlap,
                           min_chunk_size, seps)
    _report_progress(on_progress, 4, 4)
    return chunks


# ══════════════════════════════════════════════════════════════════════
# 内部辅助函数
# ══════════════════════════════════════════════════════════════════════

def _report_progress(callback: Callable[[int, int], None] | None,
                     current: int, total: int) -> None:
    """安全调用进度回调。"""
    if callback is not None:
        try:
            callback(current, total)
        except Exception:
            pass


def _read_with_fallback(path: Path, fallbacks: list[str]) -> str:
    """按优先级尝试多种编码读取文件。"""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        pass
    for enc in fallbacks:
        try:
            return path.read_text(encoding=enc, errors="replace")
        except (UnicodeDecodeError, LookupError):
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def _normalize_whitespace(text: str) -> str:
    """统一换行符、合并连续空行、去除首尾空白。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _merge_short_paragraphs(raw_paragraphs: list[str], min_size: int,
                            section_pattern: str | None = None) -> list[str]:
    """合并过短段落到相邻段落，节标题保留独立。"""
    if min_size <= 0:
        return [p.strip() for p in raw_paragraphs if p.strip()]

    sec_re = _re.compile(section_pattern, _re.IGNORECASE) if section_pattern else None

    def _is_sec_header(t: str) -> bool:
        return sec_re is not None and len(t) < 120 and bool(sec_re.match(t.strip()))

    paragraphs: list[str] = []
    buf: list[str] = []

    def _flush(target: str | None = None):
        nonlocal buf
        if not buf:
            return
        merged = " ".join(buf)
        if target and paragraphs and paragraphs[-1] == target:
            paragraphs[-1] = target + "\n" + merged
        elif target:
            paragraphs.append(target + "\n" + merged)
        elif paragraphs:
            paragraphs[-1] += "\n" + merged
        else:
            paragraphs.append(merged)
        buf = []

    for para in raw_paragraphs:
        cleaned = para.strip()
        if not cleaned:
            continue
        if _is_sec_header(cleaned):
            _flush()
            paragraphs.append(cleaned)
        elif len(cleaned) < min_size:
            buf.append(cleaned)
        else:
            _flush(cleaned) if buf else paragraphs.append(cleaned)

    if buf:
        _flush()
    return paragraphs


def _build_section_pattern() -> str:
    """构建节标题检测正则。"""
    return "(?:" + "|".join([
        r"(?:Section|CHAPTER|Chapter|Part)\s+[\dIVX]+[\.\:]?",
        r"第[一二三四五六七八九十\d]+[章节]",
        r"[一二三四五六七八九十]+[、．.]",
        r"\d+\.[\t ]*[A-Z][a-z]+",
        r"\b(?:Abstract|References?|Acknowledgments?|Appendix|Appendices|"
        r"Introduction|Conclusion|Summary|Methodology|"
        r"Related\s+Work|Background|Future\s+Work)\b",
    ]) + ")"


# ── 句子切分与缩写处理 ──────────────────────────────────────────
_SENTENCE_SPLIT_PATTERN: _re.Pattern | None = None

_KNOWN_ABBREVIATIONS: set[str] = {
    "mr", "mrs", "ms", "dr", "prof", "gen", "rep", "sen", "st",
    "vs", "etc", "cf", "al", "vol", "pp",
    "fig", "eq", "ref", "sec",
    "approx", "dept", "est", "govt", "univ",
    "i.e", "e.g", "a.k.a", "no", "nos",
}


def _get_sentence_splitter() -> _re.Pattern:
    """获取句子切分正则（中文标点、英文句号+大写/中文、换行）。"""
    global _SENTENCE_SPLIT_PATTERN
    if _SENTENCE_SPLIT_PATTERN is None:
        _SENTENCE_SPLIT_PATTERN = _re.compile(
            r"(?<=[。！？])\s*|(?<=[.!?])(?=\s+[A-Z\u4e00-\u9fff]|\n|\s*$)|(?<=\n)\s*")
    return _SENTENCE_SPLIT_PATTERN


def _merge_abbreviation_splits(sentences: list[str]) -> list[str]:
    """合并被误切的缩写片段（如 e.g., i.e., Fig. 1）。"""
    if len(sentences) <= 1:
        return sentences
    merged: list[str] = []
    for sent in sentences:
        stripped = sent.strip()
        if not stripped:
            continue
        if merged:
            prev = merged[-1].rstrip()
            prev_lower = prev.lower()
            ends_abbrev = any(prev_lower.endswith(" " + a) or prev_lower.endswith(a)
                              for a in _KNOWN_ABBREVIATIONS)
            ends_single = _re.search(r"(?<!\b[A-Z][a-z])\b[A-Z]\.$", prev) is not None
            if ends_abbrev or ends_single:
                merged[-1] = prev + " " + stripped
                continue
        merged.append(stripped)
    return merged


def _split_long_paragraph(text: str, max_size: int) -> list[str]:
    """将超长段落按句子边界切分，超长句在单词边界硬截断。"""
    raw = [s for s in _get_sentence_splitter().split(text) if s.strip()]
    sentences = _merge_abbreviation_splits(raw)

    parts: list[str] = []
    current: str = ""
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        if len(s) > max_size:
            if current:
                parts.append(current.strip())
                current = ""
            parts.extend(_hard_split_at_word_boundary(s, max_size))
        elif not current:
            current = s
        elif len(current) + len(s) <= max_size:
            current += " " + s
        else:
            parts.append(current.strip())
            current = s
    if current.strip():
        parts.append(current.strip())
    return parts


def _hard_split_at_word_boundary(text: str, max_size: int) -> list[str]:
    """按 max_size 截断，在空格/标点/换行处断开。"""
    result: list[str] = []
    remaining = text
    while len(remaining) > max_size:
        best_cut = max_size
        for cut in range(max_size, max(max_size - 100, max_size // 2), -1):
            ch = remaining[cut - 1] if cut > 0 else ""
            if ch in (" ", "\n") or (ch in "。！？.!?；;：:，，" and
               (cut >= len(remaining) or remaining[cut] not in (" ", "\n"))):
                best_cut = cut
                break
        result.append(remaining[:best_cut].strip())
        remaining = remaining[best_cut:].strip()
    if remaining:
        result.append(remaining)
    return result


def _build_chunks(paragraphs: list[str], max_chunk_size: int,
                  chunk_overlap: int, min_chunk_size: int,
                  section_pattern: str | None) -> list[str]:
    """将段落组装为文本块，支持重叠窗口和节边界感知。"""
    sec_re = _re.compile(section_pattern, _re.IGNORECASE) if section_pattern else None

    chunks: list[str] = []
    current: str = ""
    overlap_prefix: str = ""

    def _finalize() -> str:
        nonlocal current, overlap_prefix
        chunk = current.strip()
        if chunk_overlap > 0 and chunk:
            overlap_prefix = (chunk[-chunk_overlap:] if len(chunk) > chunk_overlap
                              else chunk)
        current = ""
        return chunk

    for para in paragraphs:
        ps = para.strip()
        if not ps:
            continue

        is_sec = (sec_re is not None and len(ps) < 120
                  and sec_re.match(ps))
        if is_sec and current:
            chunks.append(_finalize())

        if len(ps) > max_chunk_size:
            if current:
                chunks.append(_finalize())
            for part in _split_long_paragraph(ps, max_chunk_size):
                pw = (overlap_prefix + "\n\n" + part) if overlap_prefix else part
                overlap_prefix = ""
                if len(pw) <= max_chunk_size:
                    if not current:
                        current = pw
                    elif len(current) + len("\n\n") + len(part) <= max_chunk_size:
                        current += "\n\n" + part
                    else:
                        chunks.append(_finalize())
                        current = part
                else:
                    if current:
                        chunks.append(_finalize())
                    chunks.append(pw[:max_chunk_size].strip())
        else:
            if not current:
                current = (overlap_prefix + "\n\n" + ps) if overlap_prefix else ps
                overlap_prefix = ""
            elif len(current) + 2 + len(ps) <= max_chunk_size:
                current += "\n\n" + ps
            else:
                chunks.append(_finalize())
                current = (overlap_prefix + "\n\n" + ps) if overlap_prefix else ps
                overlap_prefix = ""

    if current.strip():
        chunks.append(_finalize())
    if min_chunk_size > 0:
        chunks = _merge_small_chunks(chunks, min_chunk_size, max_chunk_size)
    return chunks


def _merge_small_chunks(chunks: list[str], min_size: int,
                        max_size: int) -> list[str]:
    """将过短 chunk 合并到前一 chunk。"""
    if len(chunks) <= 1:
        return chunks
    merged: list[str] = []
    for chunk in chunks:
        if not merged or len(chunk) >= min_size:
            merged.append(chunk)
        else:
            combined = merged[-1] + "\n\n" + chunk
            if len(combined) <= max_size:
                merged[-1] = combined
            else:
                merged.append(chunk)
    return merged


# ── 简易自测（仅在直接运行本文件时执行）──────────────────────────
if __name__ == "__main__":
    import tempfile

    # ── 构造测试文本：包含正常段落、短段落、超长段落、节标题 ──
    sample = (
        "Section 1. Introduction\n\n"
        + "This is the first paragraph for testing chunk splitting. "
        + "It contains normal content with multiple sentences. "
        + "We need enough text to reach reasonable lengths. " + "X" * 80 + "\n\n"
        + "Short.\n\n"
        + "This is also a relatively short paragraph that should be "
        + "merged into the previous one to avoid fragmentation.\n\n"
        + "A" * 2100 + "\n\n"
        + "Section 2. Related Work\n\n"
        + "Second section with normal content. " + "Y" * 80 + "\n\n"
        + "Third paragraph continues with more content for testing. " + "Z" * 80
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(sample)
        tmp_path = f.name

    try:
        # ── 测试 1：基础分块 ─────────────────────────────────
        print("=" * 60)
        print("测试 1：基础分块 (max_chunk_size=2000)")
        print("=" * 60)
        chunks = load_and_chunk(tmp_path, max_chunk_size=2000)
        print(f"共生成 {len(chunks)} 个 chunks:\n")
        for i, c in enumerate(chunks, 1):
            print(f"--- Chunk {i} (长度: {len(c)}) ---")
            print(c[:200] + ("..." if len(c) > 200 else ""))
            print()

        # ── 测试 2：带重叠窗口 ─────────────────────────────
        print("=" * 60)
        print("测试 2：带重叠窗口 (overlap=100)")
        print("=" * 60)
        chunks_overlap = load_and_chunk(tmp_path, max_chunk_size=2000, chunk_overlap=100)
        print(f"共生成 {len(chunks_overlap)} 个 chunks\n")
        for i, c in enumerate(chunks_overlap, 1):
            print(f"--- Chunk {i} (长度: {len(c)}) ---")
            print(c[:200] + ("..." if len(c) > 200 else ""))
            print()

        # ── 测试 3：带最小 chunk 限制 ──────────────────────
        print("=" * 60)
        print("测试 3：最小 chunk 限制 (min_chunk_size=100)")
        print("=" * 60)
        chunks_min = load_and_chunk(tmp_path, max_chunk_size=2000, min_chunk_size=100)
        print(f"共生成 {len(chunks_min)} 个 chunks\n")

        # ── 统计信息 ────────────────────────────────────────
        print("=" * 60)
        print("统计信息")
        print("=" * 60)
        sizes = [len(c) for c in chunks]
        if sizes:
            print(f"  Chunk 数量:   {len(sizes)}")
            print(f"  最小:         {min(sizes)} 字符")
            print(f"  最大:         {max(sizes)} 字符")
            print(f"  平均:         {sum(sizes) / len(sizes):.0f} 字符")
            print(f"  总字符数:     {sum(sizes)}")

    finally:
        Path(tmp_path).unlink()
