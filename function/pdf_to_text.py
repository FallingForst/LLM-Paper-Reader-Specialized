"""
PDF 转文本工具

提供 transfer_to_text 函数，将 PDF 文档转换为纯文本文件，
输出格式与 chunk.load_and_chunk 兼容（段落间以空行分隔）。
"""

import re
from pathlib import Path


# ── 编号 / 项目符号正则 ──────────────────────────────────────────
_RE_NUMBERED = re.compile(
    r'^('
    # 阿拉伯数字编号: "1.", "1.1", "(1)", "1)", "[1]", "1 "
    r'[\(\[]?\d+[\.\)\]][ \t]'
    r'|[\d]+\.[\d]+[\.]?[ \t]?'
    # 罗马数字: "I.", "IV.", "viii."
    r'|[IVXLCDM]+\.'
    # 拉丁字母编号: "A.", "a)", "A "
    r'|[A-Fa-f][\.\)][ \t]'
    # 中文序号: "一、", "二.", "（三）"
    r'|[一二三四五六七八九十]+[、．.\s]'
    r'|[（\(][一二三四五六七八九十]+[）\)]'
    # 图表标题: "Figure 1.", "Table 2:"
    r'|(?:Figure|Fig\.?|Table|Tab\.?)\s*\d+[\.:]'
    # 章节标题: "Section 1", "CHAPTER 2", "第X章"
    r'|(?:Section|CHAPTER|Chapter|Part)\s+\d+'
    r'|第[一二三四五六七八九十\d]+[章节]'
    r')'
)

_RE_BULLET = re.compile(
    r'^[•●○▪▸►▻◆◇■□➢➤–—\-–—\*]\s'
)

# 独立页码/页眉模式（用于过滤）
_RE_PAGE_NUM = re.compile(r'^\d{1,4}$')
_RE_ALL_CAPS = re.compile(r'^[A-Z][A-Z\s\-–—]{8,}$')


def transfer_to_text(file_path: str, output_path: str) -> None:
    """将 PDF 文档转换为 load_and_chunk 可识别的纯文本文件。

    提取 PDF 中的文本内容，规范化段落格式（段落间以空行分隔），
    并保存为 UTF-8 编码的纯文本文件，供 load_and_chunk 进一步分块。

    Args:
        file_path: 输入的 PDF 文件路径。
        output_path: 输出的纯文本文件路径（.txt）。

    Raises:
        FileNotFoundError: 当输入的 PDF 文件不存在时。
        ValueError: 当输入文件不是 PDF 格式时。
        ImportError: 当未安装 pdfplumber 库时。
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "需要安装 pdfplumber 库。请执行: pip install pdfplumber"
        )

    input_path = Path(file_path)
    out_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {file_path}")
    if input_path.suffix.lower() != ".pdf":
        raise ValueError(f"输入文件不是 PDF 格式: {file_path}")

    # ── 1. 提取 PDF 文本 ─────────────────────────────────────
    all_paragraphs: list[str] = []

    with pdfplumber.open(str(input_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            # 将页面内的文本按行拆分，合并为段落
            lines = text.split("\n")
            page_paragraphs: list[str] = []
            current_para: str = ""
            prev_line_raw: str = ""  # 上一行原始文本（含缩进）

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    # 空行 → 段落边界
                    if current_para:
                        page_paragraphs.append(current_para)
                        current_para = ""
                    prev_line_raw = ""
                else:
                    indent = len(line) - len(line.lstrip())
                    prev_indent = len(prev_line_raw) - len(prev_line_raw.lstrip()) if prev_line_raw else 0

                    # 判断是否是新段落起点
                    if current_para and _is_new_paragraph_start(
                        stripped, current_para,
                        line_indent=indent, prev_indent=prev_indent,
                        prev_line_raw=prev_line_raw,
                    ):
                        page_paragraphs.append(current_para)
                        current_para = stripped
                    else:
                        if current_para:
                            current_para += " " + stripped
                        else:
                            current_para = stripped

                prev_line_raw = line

            if current_para:
                page_paragraphs.append(current_para)

            all_paragraphs.extend(page_paragraphs)

    # ── 2. 后处理：去重、过滤页眉页码、合并过短段落 ──────
    merged: list[str] = []
    for para in all_paragraphs:
        cleaned = para.strip()
        if not cleaned:
            continue
        # 过滤独立页码和页眉
        if _is_page_number_or_header(cleaned):
            continue
        # 合并过短碎片（如被误拆的句子片段）
        if len(cleaned) < 30 and merged:
            merged[-1] += " " + cleaned
        else:
            merged.append(cleaned)

    # ── 3. 以双换行分隔写入 ─────────────────────────────────
    output_text = "\n\n".join(merged)
    out_path.write_text(output_text, encoding="utf-8")


def _is_new_paragraph_start(
    line: str,
    prev_para: str,
    line_indent: int = 0,
    prev_indent: int = 0,
    prev_line_raw: str = "",
) -> bool:
    """启发式判断当前行是否为新段落的起始。

    综合以下信号：
    1. 上一段以句子结束符结尾 + 当前行以大写字/中文开头
    2. 当前行匹配编号或项目符号模式
    3. 缩进突变（当前行缩进显著不同于上一行）
    4. 当前行为全大写短标题

    Args:
        line: 当前行（已 strip）。
        prev_para: 上一段累积文本（已 strip）。
        line_indent: 当前行原始缩进空格数。
        prev_indent: 上一行原始缩进空格数。
        prev_line_raw: 上一行原始文本（含缩进）。
    """
    if not line:
        return False

    prev_para = prev_para.rstrip()

    # ── 信号 1：上一段以句子结束符结尾 ─────────────────────
    ends_with_punct = prev_para.endswith(
        (".", "。", "！", "？", "!", "?", ":", "：", ";", "；")
    )

    # ── 信号 2：当前行以大写字母 / 中文开头 ────────────────
    starts_capital_or_cjk = bool(
        line[0].isupper()
        or "\u4e00" <= line[0] <= "\u9fff"
        or "\u3000" <= line[0] <= "\u303f"
    )

    # ── 信号 3：当前行匹配编号模式 ─────────────────────────
    is_numbered = bool(_RE_NUMBERED.match(line))

    # ── 信号 4：当前行匹配项目符号模式 ─────────────────────
    is_bullet = bool(_RE_BULLET.match(line))

    # ── 信号 5：缩进显著增大（新段落/子项起始）─────────────
    # PDF 中缩进通常 ≥ 2 个空格视为有意义缩进
    indent_jump = (
        line_indent >= 4 and prev_indent >= 0 and line_indent > prev_indent + 1
    )

    # ── 信号 6：上一行为短行且可能为标题 ────────────────────
    prev_is_short_title = (
        len(prev_para) < 60
        and _RE_ALL_CAPS.match(prev_para) is not None
    )

    # ── 信号 7：当前行全大写（可能为新章节标题）─────────────
    is_all_caps_title = _RE_ALL_CAPS.match(line) is not None

    # ── 组合判断 ───────────────────────────────────────────
    # 编号 / 项目符号几乎总是新段落
    if is_numbered or is_bullet:
        return True

    # 全大写短标题几乎总是新段落
    if is_all_caps_title:
        return True

    # 缩进突变 + 大写/中文开头 → 新段落
    if indent_jump and starts_capital_or_cjk:
        return True

    # 上一段以标点结束 + 当前行是典型段落开头
    if ends_with_punct and (starts_capital_or_cjk or is_numbered):
        return True

    # 上一行是短标题 → 新段落
    if prev_is_short_title and starts_capital_or_cjk:
        return True

    return False


def _is_page_number_or_header(line: str) -> bool:
    """判断是否为孤立页码或页眉。"""
    stripped = line.strip()
    if not stripped:
        return True
    if _RE_PAGE_NUM.match(stripped):
        return True
    # 过短且全大写的行通常是页眉
    if len(stripped) < 20 and _RE_ALL_CAPS.match(stripped):
        return True
    return False
