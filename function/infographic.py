"""信息图生成工具 —— 使用 matplotlib 将结构化数据渲染为 PNG 信息图。"""

import textwrap
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

matplotlib.use("Agg")

# ── 配色方案 ──────────────────────────────────────────────────────
BG_COLOR = "#F5F5F5"
TITLE_COLOR = "#1A1A2E"
SUBTITLE_COLOR = "#444444"
CARD_COLORS = {
    "Hardware":   {"bg": "#D6EAF8", "border": "#2980B9", "text": "#1A5276"},
    "Algorithm":  {"bg": "#D5F5E3", "border": "#27AE60", "text": "#1E8449"},
    "Application": {"bg": "#FDEBD0", "border": "#E67E22", "text": "#A04000"},
}
TIMELINE_LEFT_COLOR = "#AED6F1"
TIMELINE_RIGHT_COLOR = "#F9E79F"
TIMELINE_LEFT_BORDER = "#2E86C1"
TIMELINE_RIGHT_BORDER = "#D4AC0D"
ARROW_COLOR = "#7F8C8D"
TEXT_DARK = "#2C3E50"
TEXT_MEDIUM = "#555555"

# ── 字体设置（优先使用系统中文字体）──────────────────────────────
def _setup_fonts() -> dict[str, FontProperties]:
    """尝试加载中文字体，回退到 sans-serif。"""
    for fp_name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                     "WenQuanYi Micro Hei", "Arial Unicode MS"]:
        try:
            FontProperties(family=fp_name).get_name()
            break
        except Exception:
            fp_name = "sans-serif"

    def _fp(weight="normal"):
        return FontProperties(family=fp_name, weight=weight)

    return {"title": _fp("bold"), "subtitle": _fp(), "body": _fp(), "small": _fp()}


# ── 辅助：文字换行 ─────────────────────────────────────────────────

def _wrap_text(text: str, width: int = 80) -> list[str]:
    """将长文本按指定字符宽度换行（固定字符数，简单回退方案）。"""
    if not text:
        return [""]
    lines: list[str] = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return lines


def _measure_text_width_data(
    text: str, ax: plt.Axes,
    fontproperties: FontProperties, fontsize: float,
) -> float:
    """测量文本渲染后的实际宽度（数据坐标系单位）。"""
    if not text:
        return 0.0
    t = ax.text(0, 0, text, fontproperties=fontproperties,
                fontsize=fontsize, alpha=0, zorder=-100)
    try:
        bbox = t.get_window_extent(ax.figure.canvas.get_renderer())
        return bbox.transformed(ax.transData.inverted()).width
    finally:
        t.remove()


def _wrap_text_to_fit(
    text: str, max_width: float, ax: plt.Axes,
    fontproperties: FontProperties, fontsize: float,
) -> list[str]:
    """根据字体和可用宽度自适应换行（二分查找断点，正确处理 CJK/拉丁混排）。"""
    if not text:
        return [""]
    lines: list[str] = []
    for paragraph in text.split("\n"):
        para = paragraph.strip()
        if not para:
            lines.append("")
            continue
        while para:
            lo, hi = 1, len(para)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                w = _measure_text_width_data(para[:mid], ax, fontproperties, fontsize)
                if w <= max_width:
                    lo = mid
                else:
                    hi = mid - 1
            if lo == 0:
                lo = 1
            lines.append(para[:lo])
            para = para[lo:].lstrip()
        return lines


# ── 布局常量 ──────────────────────────────────────────────────────
CANVAS_W = 2200
TITLE_TOP = 1520
card_margin = 60
card_gap = 30
card_start_x = card_margin
card_w = (CANVAS_W - 2 * card_margin - 2 * card_gap) // 3

CARD_BAR_H = 50
CARD_TITLE_MARGIN = 20
CARD_BOTTOM_MARGIN = 10
CARD_LINE_H = 19
CARD_ITEM_GAP = 8
CARD_MIN_H = 200

TL_HEADER_H = 34
TL_TITLE_MARGIN = 30
TL_BOTTOM_MARGIN = 15
TL_LINE_H = 20
TL_ITEM_GAP = 7
TL_MIN_H = 300


def _count_lines(items: list[str], max_width: float, ax: plt.Axes,
                 fontproperties: FontProperties, fontsize: float,
                 max_items: int = 8, max_lines_per_item: int = 6) -> int:
    """预计算文本换行后的总行数。"""
    total = 0
    for item in items[:max_items]:
        wrapped = _wrap_text_to_fit(item, max_width, ax, fontproperties, fontsize)
        total += min(len(wrapped), max_lines_per_item)
    return total


def _compute_card_height(items: list[str], card_text_width: float,
                          ax: plt.Axes, fontproperties: FontProperties,
                          fontsize: float) -> float:
    """根据内容计算卡片所需最小高度。"""
    lines = _count_lines(items, card_text_width, ax, fontproperties, fontsize)
    n = min(len(items), 8)
    text_h = lines * CARD_LINE_H + max(n - 1, 0) * CARD_ITEM_GAP
    return max(CARD_BAR_H + CARD_TITLE_MARGIN + text_h + CARD_BOTTOM_MARGIN, CARD_MIN_H)


def _compute_section_height(items: list[str], section_text_width: float,
                             ax: plt.Axes, fontproperties: FontProperties,
                             fontsize: float) -> float:
    """根据内容计算时间轴区块所需最小高度。"""
    lines = _count_lines(items, section_text_width, ax, fontproperties, fontsize,
                         max_items=10, max_lines_per_item=5)
    n = min(len(items), 10)
    text_h = lines * TL_LINE_H + max(n - 1, 0) * TL_ITEM_GAP
    return max(TL_HEADER_H + TL_TITLE_MARGIN + text_h + TL_BOTTOM_MARGIN, TL_MIN_H)


def _draw_card(ax: plt.Axes, x: float, y: float, w: float, h: float,
               title: str, items: list[str],
               color_scheme: dict[str, str],
               fonts: dict[str, FontProperties]) -> None:
    """绘制圆角卡片：背景 + 顶部色条 + 标题 + 特征列表。"""
    # 背景
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.15",
        facecolor=color_scheme["bg"], edgecolor=color_scheme["border"],
        linewidth=2.5, zorder=2))
    # 顶部色条
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y + h - CARD_BAR_H), w, CARD_BAR_H,
        boxstyle="round,pad=0.08",
        facecolor=color_scheme["border"], edgecolor="none", zorder=3))
    # 标题
    ax.text(x + w / 2, y + h - CARD_BAR_H / 2, title.upper(),
            ha="center", va="center", fontproperties=fonts["title"],
            fontsize=14, color="white", zorder=4)
    # 特征列表
    text_y = y + h - CARD_BAR_H - CARD_TITLE_MARGIN
    pw = _measure_text_width_data("• ", ax, fonts["small"], fontsize=10)
    ctw = w - 28 - pw
    for item in items[:8]:
        wrapped = _wrap_text_to_fit(item, ctw, ax, fonts["small"], fontsize=10)
        for j, line in enumerate(wrapped[:6]):
            prefix = "• " if j == 0 else "  "
            ax.text(x + 12, text_y, prefix + line, ha="left", va="top",
                    fontproperties=fonts["small"], fontsize=10,
                    color=color_scheme["text"], zorder=4)
            text_y -= CARD_LINE_H
        text_y -= CARD_ITEM_GAP


def _draw_timeline_section(ax: plt.Axes, x: float, y: float, w: float, h: float,
                           label: str, items: list[str],
                           bg_color: str, border_color: str,
                           fonts: dict[str, FontProperties]) -> None:
    """绘制时间轴区块：背景 + 标题 + 分割线 + 趋势列表。"""
    # 背景
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.15",
        facecolor=bg_color, edgecolor=border_color,
        linewidth=2, zorder=2))
    # 标题
    ax.text(x + w / 2, y + h - TL_HEADER_H / 2 - 4, label,
            ha="center", va="center", fontproperties=fonts["subtitle"],
            fontsize=14, color=TEXT_DARK, zorder=4)
    # 分割线
    ax.plot([x + 30, x + w - 30],
            [y + h - TL_HEADER_H - 10, y + h - TL_HEADER_H - 10],
            color=border_color, linewidth=1, alpha=0.5, zorder=4)
    # 趋势列表
    text_y = y + h - TL_HEADER_H - TL_TITLE_MARGIN
    pw = _measure_text_width_data("› ", ax, fonts["small"], fontsize=10.5)
    ttw = w - 32 - pw
    for item in items[:10]:
        wrapped = _wrap_text_to_fit(item, ttw, ax, fonts["small"], fontsize=10.5)
        for j, line in enumerate(wrapped[:5]):
            prefix = "› " if j == 0 else "  "
            ax.text(x + 14, text_y, prefix + line, ha="left", va="top",
                    fontproperties=fonts["small"], fontsize=10.5,
                    color=TEXT_MEDIUM, zorder=4)
            text_y -= TL_LINE_H
        text_y -= TL_ITEM_GAP


# ── 主函数 ──────────────────────────────────────────────────────────

def generate_png(data: dict[str, Any], output_path: str) -> None:
    """根据结构化数据生成信息图并保存为 PNG。

    布局: 大标题 + 副标题 | Hardware/Algorithm/Application 三卡片 | 时间轴。
    data 需含: core_goal, three_layers, timeline。
    """
    fonts = _setup_fonts()

    fig, ax = plt.subplots(figsize=(22, 16), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 2200)
    ax.set_ylim(0, 1600)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    core_goal = data.get("core_goal", {})
    layers: list[dict[str, Any]] = data.get("three_layers", [])
    timeline = data.get("timeline", {})
    definition = core_goal.get("definition", "")
    short_term: list[str] = timeline.get("short_term_2_5_years", [])
    long_term: list[str] = timeline.get("long_term_6_10_years", [])

    # ── 标题区域 ──────────────────────────────────────────
    ax.text(1100, TITLE_TOP, "AI+HW 2035: 1000× Intelligence per Joule",
            ha="center", va="center", fontproperties=fonts["title"],
            fontsize=32, color=TITLE_COLOR, zorder=5)

    divider_y = TITLE_TOP - 40
    ax.plot([300, 1900], [divider_y, divider_y],
            color="#CCCCCC", linewidth=1.5, zorder=5)

    if definition:
        wrapped_def = _wrap_text(definition, width=80)
        sub_y = divider_y - 28
        for line in wrapped_def[:3]:
            ax.text(1100, sub_y, line, ha="center", va="center",
                    fontproperties=fonts["subtitle"], fontsize=13,
                    color=SUBTITLE_COLOR, zorder=5)
            sub_y -= 20

    # ── 预计算高度 ────────────────────────────────────────
    pw = _measure_text_width_data("• ", ax, fonts["small"], fontsize=10)
    card_text_width = card_w - 28 - pw

    card_heights = [_compute_card_height(
        ld.get("key_characteristics", []), card_text_width, ax,
        fonts["small"], fontsize=10) for ld in layers]
    dynamic_card_h = max(card_heights) if card_heights else CARD_MIN_H

    title_bottom_y = (divider_y - 28 - 20 * min(len(_wrap_text(definition, width=80)), 3)
                      if definition else divider_y - 40)
    card_top_y = title_bottom_y - 15
    card_y = card_top_y - dynamic_card_h
    timeline_top_y = card_y - 12

    pw2 = _measure_text_width_data("› ", ax, fonts["small"], fontsize=10.5)
    section_w = (CANVAS_W - 2 * card_margin - 260) // 2
    section_text_w = section_w - 32 - pw2
    short_h = _compute_section_height(short_term, section_text_w, ax,
                                       fonts["small"], fontsize=10.5)
    long_h = _compute_section_height(long_term, section_text_w, ax,
                                      fonts["small"], fontsize=10.5)
    dynamic_section_h = max(short_h, long_h)
    available = timeline_top_y - 50
    if dynamic_section_h > available:
        dynamic_section_h = available

    # ── 卡片 ──────────────────────────────────────────────
    for i, ld in enumerate(layers):
        name = ld.get("layer", f"Layer {i+1}")
        chars = ld.get("key_characteristics", [])
        nl = name.strip().lower()
        if "hardware" in nl:
            scheme = CARD_COLORS["Hardware"]
        elif "algorithm" in nl:
            scheme = CARD_COLORS["Algorithm"]
        elif "application" in nl:
            scheme = CARD_COLORS["Application"]
        else:
            scheme = list(CARD_COLORS.values())[i % 3]
        _draw_card(ax, card_start_x + i * (card_w + card_gap), card_y,
                   card_w, dynamic_card_h, name, chars, scheme, fonts)

    # ── 时间轴 ────────────────────────────────────────────
    left_x = card_margin
    right_x = left_x + section_w + 260

    _draw_timeline_section(ax, left_x, 50, section_w, dynamic_section_h,
                           "近期 2—5 年", short_term,
                           TIMELINE_LEFT_COLOR, TIMELINE_LEFT_BORDER, fonts)
    _draw_timeline_section(ax, right_x, 50, section_w, dynamic_section_h,
                           "远期 6—10 年", long_term,
                           TIMELINE_RIGHT_COLOR, TIMELINE_RIGHT_BORDER, fonts)

    # 中间箭头
    arrow_cx = left_x + section_w + 130
    arrow_cy = 50 + dynamic_section_h / 2
    ax.annotate("", xy=(right_x - 12, arrow_cy),
                xytext=(left_x + section_w + 12, arrow_cy),
                arrowprops=dict(arrowstyle="->", lw=5, color=ARROW_COLOR,
                                connectionstyle="arc3,rad=0"), zorder=5)
    ax.text(arrow_cx, arrow_cy + 50, "技术演进", ha="center", va="center",
            fontproperties=fonts["small"], fontsize=13, color=ARROW_COLOR, zorder=5)

    # ── 水印与保存 ────────────────────────────────────────
    ax.text(1100, 20, "Generated by AI+HW 2035 Analyzer",
            ha="center", va="center", fontproperties=fonts["small"],
            fontsize=9, color="#BBBBBB", zorder=5)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    print(f"信息图已保存至: {output_path}")


# ── 自测 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 构造测试数据
    test_data: dict[str, Any] = {
        "core_goal": {
            "definition": (
                "实现每焦耳能量下 1000 倍智能计算效率提升，"
                "通过软硬件协同设计重塑未来 AI 基础设施"
            ),
            "current_baseline": "2025 年主流 GPU 集群能效约为 1 TOPS/W",
            "target_metric": "2035 年达到 1000 TOPS/W，实现 1000× 提升",
        },
        "three_layers": [
            {
                "layer": "Hardware",
                "key_characteristics": [
                    "3D 集成与先进封装技术",
                    "硅光子互连与近存计算",
                    "模拟存内计算与新型器件",
                ],
            },
            {
                "layer": "Algorithm",
                "key_characteristics": [
                    "稀疏化与混合精度训练",
                    "神经架构搜索（NAS）自动化",
                    "联邦学习与隐私保护推理",
                ],
            },
            {
                "layer": "Application",
                "key_characteristics": [
                    "自动驾驶实时感知系统",
                    "大模型边缘端高效部署",
                    "科学计算与药物发现加速",
                ],
            },
        ],
        "timeline": {
            "short_term_2_5_years": [
                "Chiplet 架构在数据中心大规模推广",
                "3D 堆叠 HBM 内存带宽突破 2 TB/s",
                "稀疏计算支持进入主流 AI 框架",
                "800G 硅光子互连方案开始试产",
            ],
            "long_term_6_10_years": [
                "光电混合计算成为数据中心主流",
                "存内计算芯片实现商用部署",
                "自监督学习大幅降低标注依赖",
                "神经形态计算达到实用化门槛",
            ],
        },
    }

    output = Path(__file__).parent.parent / "output" / "AI_HW_2035_infographic.png"
    generate_png(test_data, str(output))
