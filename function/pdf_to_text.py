"""PDF 转文本工具 —— 提取 PDF 正文，自动过滤页眉/注释/参考文献，输出双换行分隔的纯文本。"""

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

# ── 参考文献相关正则 ──────────────────────────────────────────
# 参考文献章节标题
_RE_REF_HEADER = re.compile(
    r'^('
    # 英文: "References", "REFERENCES", "References and Notes"
    r'[Rr]eferences?\s*$'
    r'|REFERENCES?\s*$'
    r'|[Rr]eferences?\s+and\s+[Nn]otes\s*$'
    r'|REFERENCES?\s+AND\s+NOTES\s*$'
    # 英文: "Bibliography"
    r'|[Bb]ibliography\s*$'
    r'|BIBLIOGRAPHY\s*$'
    # 英文: "Literature Cited", "Works Cited"
    r'|[Ll]iterature\s+[Cc]ited\s*$'
    r'|[Ww]orks?\s+[Cc]ited\s*$'
    # 中文: "参考文献", "引用文献"
    r'|参考文献\s*$'
    r'|引用文献\s*$'
    # 编号参考文献: "References [1]", "参考文献 [1]"（与首条合并的情况）
    r'|[Rr]eferences?\s+\[\d+\]'
    r'|参考文献\s*\[\d+\]'
    r')'
)

# 参考文献条目特征：以 [N] 或 N. 开头
_RE_REF_ENTRY_START = re.compile(
    r'^('
    r'\[\d+\]'      # [1], [23]
    r'|\d+\.\s'     # 1. , 23. 
    r')'
)

# 参考文献条目中常见的特征词/模式
_RE_REF_FEATURES = re.compile(
    r'('
    r'\bet\s+al\.?'              # et al.
    r'|\bvol\.?\s*\d+'           # vol. 10
    r'|\bpp\.?\s*\d+'            # pp. 100
    r'|\bno\.?\s*\d+'            # no. 2
    r'|\bJournal\b'              # Journal of ...
    r'|\bProceedings\b'          # Proceedings of ...
    r'|\bConference\b'           # Conference on ...
    r'|\bSymposium\b'            # Symposium
    r'|\bIEEE\b'                 # IEEE
    r'|\bACM\b'                  # ACM
    r'|\bSpringer\b'             # Springer
    r'|\bElsevier\b'             # Elsevier
    r'|\bTrans(?:actions)?\.'    # Trans. / Transactions
    r'|\b[Tt]echnical\s+[Rr]eport\b'  # Technical Report
    r'|\b[Aa]r[Xx]iv\s*[:\d]'   # arXiv:1234
    r'|\bdoi\s*[:\s]'           # doi: / doi 
    r'|\(\d{4}\)'               # (2020)
    r'|\bpp\.\s*\d+[-–]\d+'     # pp. 100-110
    r'|\bpages?\s+\d+[-–]\d+'   # pages 100-110
    r'|\b\d{4}\b'               # 年份
    r')',
    re.IGNORECASE,
)

# 独立页码/页眉模式（用于过滤）
_RE_PAGE_NUM = re.compile(r'^\d{1,4}$')
_RE_ALL_CAPS = re.compile(r'^[A-Z][A-Z\s\-–—]{8,}$')

# ── 注释 / 脚注 / 元信息过滤正则 ──────────────────────────────
_RE_COMMENT_LINE = re.compile(
    r'^('
    # 版权声明: © 2023, Copyright 2023, All rights reserved
    r'©\s*\d{4}'
    r'|Copyright\s*[©\d]'
    r'|All\s+[Rr]ights\s+[Rr]eserved'
    # 通讯作者 / 作者联系方式
    r'|[\*†‡§¶#]+\s*(Corresponding|To\s+whom|Author|E-?mail)'
    r'|(Corresponding|Correspondence)\s+(author|to)\s*[:\s]'
    r'|E-?mail\s*[:\s]\s*\S+@\S+'
    # URL / DOI 独立行
    r'|https?://\S+'
    r'|doi\s*[:\s]\s*\S+'
    r'|10\.\d{4,}/\S+'
    # 发表/接收/修订日期行
    r'|(Received|Accepted|Published|Submitted|Revised)\s*[:\s]\s*\d'
    # 期刊/出版方元信息
    r'|(Published\s+by|Publisher\s*[:\s]|Volume\s+\d+\s*[,，]|Issue\s+\d+)'
    # 独立脚注标记: * † ‡ § ¶ # ① ② 等开头且较短的行
    r'|^[\*†‡§¶#①②③④⑤⑥⑦⑧⑨⑩]\s'
    # arXiv / preprint 标识
    r'|arXiv\s*[:\s]'
    r'|Preprint\s+submitted'
    # 纯页码引用: "[1]", "[1,2]", "[1-5]" 的独立短行
    r'|^\[[\d,\s\-–—]+\]$'
    # 作者机构邮箱行: 以邮箱结尾的短行
    r'|\S+@\S+\.\S+'
    r')'
)


def transfer_to_text(file_path: str, output_path: str) -> None:
    """将 PDF 转为纯文本，自动过滤页眉/注释/参考文献并规范化段落格式。"""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("需要安装 pdfplumber 库。请执行: pip install pdfplumber")

    input_path = Path(file_path)
    out_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {file_path}")
    if input_path.suffix.lower() != ".pdf":
        raise ValueError(f"输入文件不是 PDF 格式: {file_path}")

    all_paragraphs: list[str] = []
    with pdfplumber.open(str(input_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            lines = text.split("\n")
            page_paragraphs: list[str] = []
            current_para: str = ""
            prev_line_raw: str = ""

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    if current_para:
                        page_paragraphs.append(current_para)
                        current_para = ""
                    prev_line_raw = ""
                else:
                    indent = len(line) - len(line.lstrip())
                    prev_indent = (len(prev_line_raw) - len(prev_line_raw.lstrip())
                                   if prev_line_raw else 0)
                    if current_para and _is_new_paragraph_start(
                        stripped, current_para, line_indent=indent,
                        prev_indent=prev_indent, prev_line_raw=prev_line_raw,
                    ):
                        page_paragraphs.append(current_para)
                        current_para = stripped
                    else:
                        current_para = (current_para + " " + stripped
                                        if current_para else stripped)
                prev_line_raw = line

            if current_para:
                page_paragraphs.append(current_para)
            all_paragraphs.extend(page_paragraphs)

    # ── 过滤：移除参考文献及之后内容、页眉/注释 ──────────
    ref_start_idx: int | None = None
    for i, para in enumerate(all_paragraphs):
        if _is_reference_header(para):
            ref_start_idx = i
            break
    if ref_start_idx is None:
        consecutive = 0
        for i, para in enumerate(all_paragraphs):
            if _is_reference_entry(para):
                consecutive += 1
                if consecutive >= 4:
                    ref_start_idx = i - consecutive + 1
                    break
            else:
                consecutive = 0

    merged: list[str] = []
    for idx, para in enumerate(all_paragraphs):
        if ref_start_idx is not None and idx >= ref_start_idx:
            continue
        cleaned = para.strip()
        if not cleaned:
            continue
        if _is_page_number_or_header(cleaned):
            continue
        if _is_comment_or_footnote(cleaned):
            continue
        if len(cleaned) < 30 and merged:
            merged[-1] += " " + cleaned
        else:
            merged.append(cleaned)

    out_path.write_text("\n\n".join(merged), encoding="utf-8")


def _is_new_paragraph_start(line: str, prev_para: str,
                             line_indent: int = 0, prev_indent: int = 0,
                             prev_line_raw: str = "") -> bool:
    """启发式判断是否为新段落起始（标点结束/编号/缩进/全大写标题）。"""
    if not line:
        return False
    prev_para = prev_para.rstrip()

    ends_with_punct = prev_para.endswith(
        (".", "。", "！", "？", "!", "?", ":", "：", ";", "；"))
    starts_cjk = bool(line[0].isupper()
                      or "\u4e00" <= line[0] <= "\u9fff"
                      or "\u3000" <= line[0] <= "\u303f")
    is_numbered = bool(_RE_NUMBERED.match(line))
    is_bullet = bool(_RE_BULLET.match(line))
    indent_jump = (line_indent >= 4 and line_indent > prev_indent + 1)
    prev_is_short_title = (len(prev_para) < 60
                           and _RE_ALL_CAPS.match(prev_para) is not None)
    is_all_caps = _RE_ALL_CAPS.match(line) is not None

    if is_numbered or is_bullet or is_all_caps:
        return True
    if indent_jump and starts_cjk:
        return True
    if ends_with_punct and (starts_cjk or is_numbered):
        return True
    if prev_is_short_title and starts_cjk:
        return True
    return False


def _is_reference_header(line: str) -> bool:
    """判断是否为参考文献章节标题（≤80 字符的限制）。"""
    stripped = line.strip()
    return bool(stripped) and len(stripped) <= 80 and bool(_RE_REF_HEADER.match(stripped))


def _is_reference_entry(line: str) -> bool:
    """判断是否为参考文献条目（以 [N]/N. 开头 + 特征词）。"""
    stripped = line.strip()
    return (bool(stripped) and _RE_REF_ENTRY_START.match(stripped)
            and 20 <= len(stripped) <= 500
            and bool(_RE_REF_FEATURES.search(stripped)))


def _is_page_number_or_header(line: str) -> bool:
    """判断是否为孤立页码或页眉。"""
    stripped = line.strip()
    return (not stripped or _RE_PAGE_NUM.match(stripped)
            or (len(stripped) < 20 and _RE_ALL_CAPS.match(stripped)))


def _is_comment_or_footnote(line: str) -> bool:
    """判断是否为注释/脚注/元信息（版权、Email、URL、日期、脚注标记等）。"""
    stripped = line.strip()
    if not stripped:
        return True
    if _RE_COMMENT_LINE.match(stripped):
        return True
    if re.match(
        r'^(Author\s*[\'’]?s?\s*(contributions|information|details|affiliation)'
        r'|Conflicts?\s+of\s+interest|Competing\s+interests|Acknowledgments?'
        r'|Data\s+availability|Funding\s*[:\s]|Supplementary\s+materials?)',
        stripped, re.IGNORECASE,
    ) and len(stripped) < 120:
        return True
    return False
