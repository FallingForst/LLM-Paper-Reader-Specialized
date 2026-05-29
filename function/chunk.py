"""
文本分块工具（优化版）

提供 load_and_chunk 函数，读取纯文本文件并按指定大小分块，
支持重叠窗口、节边界感知、进度回调等高级特性，
适用于后续的文本分析、嵌入或检索增强生成（RAG）等场景。

优化要点：
  - 新增 chunk_overlap 支持，保证上下文连续性
  - 新增 min_chunk_size / min_paragraph_size 可配置阈值
  - 节标题检测，尽量在 Section 边界处切分
  - 改进句子切分：处理缩写、小数、省略号等边缘情况
  - 超长句硬截断时尽量在单词边界处断开
  - 更完善的编码回退链（UTF-8 → GBK → Latin-1）
  - 进度回调支持，便于长文本处理时展示进度
"""

from pathlib import Path
from typing import Callable


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
    """读取纯文本论文，按段落拆分并组合为不超过指定大小的文本块。

    处理流程：
    1. 读取文件内容（UTF-8 → GBK → Latin-1 编码回退）。
    2. 规范化空白与换行，按双换行符拆分为段落。
    3. 将过短段落（< min_paragraph_size 字符）合并到相邻段落，避免碎片化。
    4. 以 max_chunk_size 为上限，将段落组合为文本块；
       可选地在 chunk 之间保留重叠窗口（chunk_overlap 字符）。
    5. 可选：检测节标题（Section / 第X章 等），尽量在节边界处新开 chunk。

    Args:
        file_path: 纯文本文件的路径。
        max_chunk_size: 每个文本块的最大字符数，默认为 2000。
        chunk_overlap: 相邻 chunk 之间的重叠字符数，0 表示不重叠。
                       用于 RAG 场景保证检索时上下文连续性。
        min_chunk_size: 每个文本块的最小字符数。小于此值的末尾 chunk
                        将被合并到前一 chunk（0 表示不限制）。
        min_paragraph_size: 短段落合并阈值（字符数），默认 50。
        preserve_sections: 是否尽量在节标题边界处切分，默认 True。
        encoding_fallbacks: 编码回退列表，默认 ["gbk", "latin-1"]。
        on_progress: 进度回调 (current_step, total_steps) → None。

    Returns:
        字符串列表，每个元素为一个文本块。

    Raises:
        FileNotFoundError: 当指定路径的文件不存在时。
        ValueError: 当路径不是文件或参数无效时。
    """
    # ── 参数校验 ──────────────────────────────────────────────
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

    # ── 1. 读取文件 ──────────────────────────────────────────
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    if not path.is_file():
        raise ValueError(f"路径不是文件: {file_path}")

    text = _read_with_fallback(path, encoding_fallbacks)
    _report_progress(on_progress, 1, 4)

    # ── 2. 规范化并按空行拆分段落 ────────────────────────────
    # 统一换行符、合并多个连续空行
    text = _normalize_whitespace(text)
    raw_paragraphs = text.split("\n\n")
    _report_progress(on_progress, 2, 4)

    # ── 3. 合并过短段落 ──────────────────────────────────────
    seps = _build_section_pattern() if preserve_sections else None
    paragraphs = _merge_short_paragraphs(raw_paragraphs, min_paragraph_size,
                                          section_pattern=seps)
    _report_progress(on_progress, 3, 4)

    # ── 4. 按 max_chunk_size 组合段落为 chunks ────────────────
    chunks = _build_chunks(
        paragraphs=paragraphs,
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        section_pattern=seps,
    )
    _report_progress(on_progress, 4, 4)

    return chunks


# ══════════════════════════════════════════════════════════════════════
# 内部辅助函数
# ══════════════════════════════════════════════════════════════════════

def _report_progress(
    callback: Callable[[int, int], None] | None,
    current: int,
    total: int,
) -> None:
    """安全调用进度回调。"""
    if callback is not None:
        try:
            callback(current, total)
        except Exception:
            pass  # 进度回调不应中断主流程


def _read_with_fallback(path: Path, fallbacks: list[str]) -> str:
    """按优先级尝试多种编码读取文件。"""
    # 首选 UTF-8
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        pass

    # 依次尝试回退编码
    for enc in fallbacks:
        try:
            return path.read_text(encoding=enc, errors="replace")
        except (UnicodeDecodeError, LookupError):
            continue

    # 终极回退：二进制模式 +  surrogateescape
    raw = path.read_bytes()
    return raw.decode("utf-8", errors="replace")


def _normalize_whitespace(text: str) -> str:
    """规范化文本空白：统一换行符、合并连续空行、去除首尾空白。"""
    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 合并 3 个及以上连续换行为双换行（段落分隔）
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去除段落内部的孤立换行两侧多余空白
    return text.strip()


def _merge_short_paragraphs(
    raw_paragraphs: list[str],
    min_size: int,
    section_pattern: str | None = None,
) -> list[str]:
    """合并过短段落到相邻段落，避免碎片化。

    策略：
    - 优先向前合并（并入前一段）
    - 节标题（匹配 section_pattern）始终保留为独立段落
    - 同时合并连续多个过短段落
    """
    if min_size <= 0:
        # 不过滤，仅去除空段落
        return [p.strip() for p in raw_paragraphs if p.strip()]

    if section_pattern:
        section_re = _re.compile(section_pattern, _re.IGNORECASE)
    else:
        section_re = None

    def _is_section_header(text: str) -> bool:
        """判断是否为节标题。"""
        if section_re is None:
            return False
        return len(text) < 120 and bool(section_re.match(text.strip()))

    paragraphs: list[str] = []
    # 暂存连续短段落，等待下一个够长的段落到来时一起处理
    buffer: list[str] = []

    def _flush_buffer(target_para: str | None) -> None:
        """将缓冲区中的短段落合并到目标段落。"""
        nonlocal buffer
        if not buffer:
            return
        merged_short = " ".join(buffer)
        if target_para is not None:
            if paragraphs and paragraphs[-1] == target_para:
                paragraphs[-1] = target_para + "\n" + merged_short
            else:
                paragraphs.append(target_para + "\n" + merged_short)
        elif paragraphs:
            paragraphs[-1] += "\n" + merged_short
        else:
            paragraphs.append(merged_short)
        buffer = []

    for para in raw_paragraphs:
        cleaned = para.strip()
        if not cleaned:
            continue

        # 节标题 → 始终保留为独立段落
        if _is_section_header(cleaned):
            _flush_buffer(None)  # 先清空缓冲
            paragraphs.append(cleaned)
            continue

        if len(cleaned) < min_size:
            buffer.append(cleaned)
        else:
            if buffer:
                _flush_buffer(cleaned)
                buffer = []
            else:
                paragraphs.append(cleaned)

    # 末尾剩余的短段落：向前合并到最后一段
    if buffer:
        _flush_buffer(None)

    return paragraphs


def _build_section_pattern() -> str:
    """构建节标题检测正则模式。

    匹配以下模式（不区分大小写）：
    - Section 1, Section 2.1, SECTION III
    - Chapter 1, CHAPTER 2
    - 第X章, 第X节
    - 一、 二、 (中文序号标题)
    - 1. Introduction, 2. Related Work (编号 + 大写单词)
    """
    patterns = [
        # 英文节标题
        r"(?:Section|CHAPTER|Chapter|Part)\s+[\dIVX]+[\.\:]?",
        # 中文章节标题
        r"第[一二三四五六七八九十\d]+[章节]",
        # 中文序号标题（一、二、…）
        r"[一二三四五六七八九十]+[、．.]",
        # 编号 + 大写单词（如 "1. Introduction"）
        r"\d+\.[\t ]*[A-Z][a-z]+",
        # Abstract / References / Appendix 等固定标题
        r"\b(?:Abstract|References?|Acknowledgments?|Appendix|Appendices|"
        r"Introduction|Conclusion|Summary|Methodology|"
        r"Related\s+Work|Background|Future\s+Work)\b",
    ]
    return "(?:" + "|".join(patterns) + ")"


# ── 全局编译一次，避免每次调用重复编译 ──────────────────────
import re as _re

_SENTENCE_SPLIT_PATTERN: _re.Pattern | None = None
_SECTION_PATTERN: _re.Pattern | None = None


def _get_sentence_splitter() -> _re.Pattern:
    """获取编译后的句子切分正则。

    在以下位置切分：
    - 中文句号/问号/感叹号（。！？）
    - 英文句号/问号/感叹号（.!?）后跟空格 + 大写字母/中文
    - 换行符

    切分后再由 _merge_abbreviation_splits 修复被误切的缩写。
    """
    global _SENTENCE_SPLIT_PATTERN
    if _SENTENCE_SPLIT_PATTERN is not None:
        return _SENTENCE_SPLIT_PATTERN

    _SENTENCE_SPLIT_PATTERN = _re.compile(
        # 中文标点后切分
        r"(?<=[。！？])\s*"
        r"|"
        # 英文句号/问号/感叹号 后跟 空格+大写字母 或 空格+中文 或 换行
        r"(?<=[.!?])(?=\s+[A-Z\u4e00-\u9fff]|\n|\s*$)"
        r"|"
        # 换行处切分
        r"(?<=\n)\s*",
    )
    return _SENTENCE_SPLIT_PATTERN


# ── 常见英文缩写（其后的句点不是句子边界）─────────────────────
_KNOWN_ABBREVIATIONS: set[str] = {
    "mr", "mrs", "ms", "dr", "prof", "gen", "rep", "sen", "st",
    "vs", "etc", "cf", "al", "vol", "pp",
    "fig", "eq", "ref", "sec",
    "approx", "dept", "est", "govt", "univ",
    "i.e", "e.g", "a.k.a",
    "no", "nos",
}


def _merge_abbreviation_splits(sentences: list[str]) -> list[str]:
    """将被误切分的缩写片段合并回完整的句子。

    例如 "e.g." 后面的内容如果被错误地当作新句子开头，则将其合并回去。
    """
    if len(sentences) <= 1:
        return sentences

    merged: list[str] = []
    for sent in sentences:
        stripped = sent.strip()
        if not stripped:
            continue

        # 检查前一句是否以已知缩写结尾（如 "i.e" / "e.g"）
        if merged:
            prev = merged[-1].rstrip()
            # 取前一句最后几个字符，判断是否为缩写形式
            prev_lower = prev.lower()
            ends_with_abbrev = any(
                prev_lower.endswith(" " + abbr) or prev_lower.endswith(abbr)
                for abbr in _KNOWN_ABBREVIATIONS
            )
            # 同时检查前一句是否以单个大写字母结尾（如 "A." "B."）
            ends_with_single_letter = _re.search(
                r"(?<!\b[A-Z][a-z])\b[A-Z]\.$", prev
            ) is not None

            if ends_with_abbrev or ends_with_single_letter:
                merged[-1] = prev + " " + stripped
                continue

        merged.append(stripped)

    return merged


def _split_long_paragraph(text: str, max_size: int) -> list[str]:
    """将超长段落按句子边界切分为不超过 max_size 的片段。

    改进：
    - 处理缩写（i.e., e.g., etc.）不被误判为句子边界
    - 超长句硬截断时尽量在单词边界处断开
    - 保留句子分隔符在原句末尾
    """
    pattern = _get_sentence_splitter()
    raw_sentences = pattern.split(text)
    raw_sentences = [s for s in raw_sentences if s.strip()]
    # 修复被误切的缩写（如 "e.g." "i.e." "Fig. 1" 等）
    sentences = _merge_abbreviation_splits(raw_sentences)

    parts: list[str] = []
    current: str = ""

    for sent in sentences:
        sent_stripped = sent.strip()
        if not sent_stripped:
            continue

        if len(sent_stripped) > max_size:
            # 单句仍超长：先保存 current，再对该句做单词边界感知截断
            if current:
                parts.append(current.strip())
                current = ""
            sub_parts = _hard_split_at_word_boundary(sent_stripped, max_size)
            parts.extend(sub_parts)
            continue

        if not current:
            current = sent_stripped
        elif len(current) + len(sent_stripped) <= max_size:
            current += " " + sent_stripped if current else sent_stripped
        else:
            parts.append(current.strip())
            current = sent_stripped

    if current.strip():
        parts.append(current.strip())

    return parts


def _hard_split_at_word_boundary(text: str, max_size: int) -> list[str]:
    """将超长文本按 max_size 截断，尽量在单词/词边界处断开。

    对于英文：在空格处断开；对于中文：在字符边界断开即可。
    """
    result: list[str] = []
    remaining = text

    while len(remaining) > max_size:
        # 试图在 max_size 以内找一个理想的断点
        chunk = remaining[:max_size]

        # 在 max_size 附近（往前最多 100 字符）找空格/标点作为断点
        search_start = max(max_size - 100, max_size // 2)
        best_cut = max_size

        for cut in range(max_size, search_start, -1):
            ch = remaining[cut - 1] if cut > 0 else ""
            next_ch = remaining[cut] if cut < len(remaining) else ""
            # 英文单词边界（空格）
            if ch == " ":
                best_cut = cut
                break
            # 中文句子边界（标点）
            if ch in "。！？.!?；;：:，," and (next_ch != " " and next_ch != "\n"):
                best_cut = cut
                break
            # 换行
            if ch == "\n":
                best_cut = cut
                break

        result.append(remaining[:best_cut].strip())
        remaining = remaining[best_cut:].strip()

    if remaining:
        result.append(remaining)

    return result


def _build_chunks(
    paragraphs: list[str],
    max_chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    section_pattern: str | None,
) -> list[str]:
    """将段落列表组装为文本块。

    核心逻辑：
    1. 遍历段落，尽量将多个段落填入同一 chunk（用双换行连接）。
    2. 单段落超过 max_chunk_size → 按句子切分。
    3. （可选）检测节标题，遇到新节时倾向于新开 chunk。
    4. 支持重叠窗口：每个 chunk 末尾 chunk_overlap 字符
       会作为下一个 chunk 的开头前缀。
    5. 末尾过短 chunk（< min_chunk_size）合并到前一个 chunk。
    """
    if section_pattern:
        section_re = _re.compile(section_pattern, _re.IGNORECASE)
    else:
        section_re = None

    chunks: list[str] = []
    current_chunk: str = ""
    overlap_prefix: str = ""  # 来自前一个 chunk 的重叠前缀

    def _finalize_chunk() -> str:
        """结束当前 chunk 并返回其内容（含重叠前缀处理）。"""
        nonlocal current_chunk, overlap_prefix
        chunk = current_chunk.strip()
        if chunk_overlap > 0 and chunk:
            # 保留末尾 overlap 字符作为下一个 chunk 的前缀
            overlap_prefix = chunk[-chunk_overlap:] if len(chunk) > chunk_overlap else chunk
        current_chunk = ""
        return chunk

    for para in paragraphs:
        para_stripped = para.strip()
        if not para_stripped:
            continue

        # ── 检测节标题：遇到新节 → 优先结束当前 chunk ──────
        is_section_header = (
            section_re is not None
            and len(para_stripped) < 120  # 标题通常较短
            and section_re.match(para_stripped)
        )

        if is_section_header and current_chunk:
            chunks.append(_finalize_chunk())

        # ── 情况 A：单段落超过 max_chunk_size → 按句子切分 ──
        if len(para_stripped) > max_chunk_size:
            if current_chunk:
                chunks.append(_finalize_chunk())

            sub_parts = _split_long_paragraph(para_stripped, max_chunk_size)
            for part in sub_parts:
                part_with_prefix = (overlap_prefix + "\n\n" + part) if overlap_prefix else part
                overlap_prefix = ""  # 前缀只用于第一个子段

                if len(part_with_prefix) <= max_chunk_size:
                    if not current_chunk:
                        current_chunk = part_with_prefix
                    elif len(current_chunk) + len("\n\n") + len(part) <= max_chunk_size:
                        current_chunk += "\n\n" + part
                    else:
                        chunks.append(_finalize_chunk())
                        current_chunk = part
                else:
                    if current_chunk:
                        chunks.append(_finalize_chunk())
                    chunks.append(part_with_prefix[:max_chunk_size].strip())
            continue

        # ── 情况 B：正常段落 → 尝试拼入当前 chunk ──────────
        if not current_chunk:
            current_chunk = (overlap_prefix + "\n\n" + para_stripped) if overlap_prefix else para_stripped
            overlap_prefix = ""
        elif len(current_chunk) + len("\n\n") + len(para_stripped) <= max_chunk_size:
            current_chunk += "\n\n" + para_stripped
        else:
            chunks.append(_finalize_chunk())
            current_chunk = (overlap_prefix + "\n\n" + para_stripped) if overlap_prefix else para_stripped
            overlap_prefix = ""

    # ── 保存最后一个未满的 chunk ────────────────────────────
    if current_chunk.strip():
        chunks.append(_finalize_chunk())

    # ── 后处理：合并过短 chunk ──────────────────────────────
    if min_chunk_size > 0:
        chunks = _merge_small_chunks(chunks, min_chunk_size, max_chunk_size)

    return chunks


def _merge_small_chunks(
    chunks: list[str],
    min_size: int,
    max_size: int,
) -> list[str]:
    """将过短的 chunk 合并到相邻 chunk，避免生成碎片化小块。"""
    if len(chunks) <= 1:
        return chunks

    merged: list[str] = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        if len(chunk) >= min_size or not merged:
            merged.append(chunk)
            i += 1
        else:
            # 当前 chunk 过短 → 尝试并入前一个 chunk
            prev = merged[-1]
            combined = prev + "\n\n" + chunk
            if len(combined) <= max_size:
                merged[-1] = combined
            else:
                # 放不进前一个 → 保留（虽然短，但别无选择）
                merged.append(chunk)
            i += 1

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
