"""
文本分块工具

提供 load_and_chunk 函数，读取纯文本文件并按指定大小分块，
适用于后续的文本分析、嵌入或检索增强生成（RAG）等场景。
"""

from pathlib import Path


def load_and_chunk(file_path: str, max_chunk_size: int = 2000) -> list[str]:
    """读取纯文本论文，按段落拆分并组合为不超过指定大小的文本块。

    处理流程：
    1. 读取文件内容（UTF-8，失败则回退到 GBK）。
    2. 按双换行符（空行）拆分为段落。
    3. 将过短段落（< 50 字符）合并到前一段，避免碎片化。
    4. 以 max_chunk_size 为上限，将段落组合为文本块。

    Args:
        file_path: 纯文本文件的路径。
        max_chunk_size: 每个文本块的最大字符数，默认为 2000。

    Returns:
        字符串列表，每个元素为一个文本块。

    Raises:
        FileNotFoundError: 当指定路径的文件不存在时。
        ValueError: 当路径不是文件时。
    """
    # ── 1. 读取文件 ──────────────────────────────────────────────
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    if not path.is_file():
        raise ValueError(f"路径不是文件: {file_path}")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # 尝试常见的中文编码（学术论文可能使用 GBK/GB2312）
        text = path.read_text(encoding="gbk", errors="replace")

    # ── 2. 按空行拆分段落 ──────────────────────────────────────
    raw_paragraphs = text.split("\n\n")

    # ── 3. 合并过短段落（< 50 字符 → 并入前一段）────────────
    paragraphs: list[str] = []
    for para in raw_paragraphs:
        cleaned = para.strip()
        if not cleaned:
            continue  # 跳过纯空行

        if len(cleaned) < 50 and paragraphs:
            # 过短段落合并到前一段
            paragraphs[-1] += "\n" + cleaned
        else:
            paragraphs.append(cleaned)

    # ── 4. 按 max_chunk_size 组合段落为 chunks ────────────────
    chunks: list[str] = []
    current_chunk: str = ""

    for para in paragraphs:
        # 4a. 如果单段落本身就超过上限，进一步按句子切分
        if len(para) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            sub_parts = _split_long_paragraph(para, max_chunk_size)
            for part in sub_parts:
                if len(part) <= max_chunk_size:
                    if not current_chunk:
                        current_chunk = part
                    elif len(current_chunk) + len("\n\n") + len(part) <= max_chunk_size:
                        current_chunk += "\n\n" + part
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = part
                else:
                    # 极端情况：单个句子仍超长，强制截断
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    chunks.append(part[:max_chunk_size].strip())
            continue

        # 4b. 正常情况：尝试将段落拼入当前 chunk
        if not current_chunk:
            current_chunk = para
        elif len(current_chunk) + len("\n\n") + len(para) <= max_chunk_size:
            current_chunk += "\n\n" + para
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para

    # 保存最后一个未满的 chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _split_long_paragraph(text: str, max_size: int) -> list[str]:
    """将超长段落按句子边界切分为不超过 max_size 的片段。

    优先在句号、问号、感叹号或换行处切分，保留分隔符在句末。
    """
    import re

    # 按中英文句子结束符 + 后续空白进行切分
    sentences = re.split(r"(?<=[。！？.!?\n])\s*", text)
    sentences = [s for s in sentences if s.strip()]

    parts: list[str] = []
    current: str = ""

    for sent in sentences:
        if len(sent) > max_size:
            # 单句仍超长：先保存 current，再硬切该句
            if current:
                parts.append(current.strip())
                current = ""
            for i in range(0, len(sent), max_size):
                parts.append(sent[i:i + max_size].strip())
            continue

        if not current:
            current = sent
        elif len(current) + len(sent) <= max_size:
            current += sent
        else:
            parts.append(current.strip())
            current = sent

    if current.strip():
        parts.append(current.strip())

    return parts


# ── 简易自测（仅在直接运行本文件时执行）──────────────────────────
if __name__ == "__main__":
    import tempfile

    # 构造测试文本：包含正常段落、短段落、超长段落
    sample = (
        "第一章 引言\n\n"
        + "这是第一段内容，用于测试段落拆分功能。" + "X" * 30 + "\n\n"
        + "短。\n\n"
        + "这也是一个比较短的段落，应该被合并到前一段中去。\n\n"
        + "A" * 2100 + "\n\n"
        + "第二段正常内容。" + "Y" * 40 + "\n\n"
        + "第三段内容，这里继续写一些东西。" + "Z" * 50
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(sample)
        tmp_path = f.name

    try:
        chunks = load_and_chunk(tmp_path, max_chunk_size=2000)
        print(f"共生成 {len(chunks)} 个 chunks:\n")
        for i, c in enumerate(chunks, 1):
            print(f"--- Chunk {i} (长度: {len(c)}) ---")
            print(c[:200] + ("..." if len(c) > 200 else ""))
            print()
    finally:
        Path(tmp_path).unlink()
