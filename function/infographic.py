"""
信息图生成工具

使用 matplotlib 将 LLM 提取的结构化数据渲染为 PNG 信息图。
"""

import textwrap
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

# 使用非交互式后端（避免 GUI 依赖）
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
    title_fp = FontProperties(family="sans-serif", weight="bold")
    subtitle_fp = FontProperties(family="sans-serif", weight="normal")
    body_fp = FontProperties(family="sans-serif", weight="normal")
    small_fp = FontProperties(family="sans-serif", weight="normal")

    # 尝试查找可用的中文字体
    for fp_name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                     "WenQuanYi Micro Hei", "Arial Unicode MS"]:
        try:
            test_fp = FontProperties(family=fp_name)
            # 简单验证：尝试获取字体名
            _ = test_fp.get_name()
            title_fp = FontProperties(family=fp_name, weight="bold")
            subtitle_fp = FontProperties(family=fp_name, weight="normal")
            body_fp = FontProperties(family=fp_name, weight="normal")
            small_fp = FontProperties(family=fp_name, weight="normal")
            break
        except Exception:
            continue

    return {
        "title": title_fp,
        "subtitle": subtitle_fp,
        "body": body_fp,
        "small": small_fp,
    }


# ── 辅助：文字换行 ─────────────────────────────────────────────────

def _wrap_text(text: str, width: int = 80) -> list[str]:
    """将长文本按指定宽度换行（按字符数）。"""
    if not text:
        return [""]
    # 对中文和英文分别处理
    lines: list[str] = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return lines


def _draw_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    items: list[str],
    color_scheme: dict[str, str],
    fonts: dict[str, FontProperties],
) -> None:
    """绘制一个圆角卡片。

    Args:
        ax: matplotlib Axes 对象。
        x, y: 卡片左下角坐标。
        w, h: 卡片宽高。
        title: 卡片标题（图层名，如 Hardware）。
        items: 关键特征列表。
        color_scheme: 配色方案字典，含 bg, border, text。
        fonts: 字体字典。
    """
    # ── 卡片背景 ──────────────────────────────────────────
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color_scheme["bg"],
        edgecolor=color_scheme["border"],
        linewidth=2.5,
        zorder=2,
    )
    ax.add_patch(rect)

    # ── 顶部色条 ──────────────────────────────────────────
    bar_h = h * 0.12
    bar = mpatches.FancyBboxPatch(
        (x, y + h - bar_h), w, bar_h,
        boxstyle="round,pad=0.08",
        facecolor=color_scheme["border"],
        edgecolor="none",
        zorder=3,
    )
    ax.add_patch(bar)

    # ── 标题 ──────────────────────────────────────────────
    ax.text(
        x + w / 2, y + h - bar_h / 2,
        title.upper(),
        ha="center", va="center",
        fontproperties=fonts["title"],
        fontsize=14,
        color="white",
        zorder=4,
    )

    # ── 特征列表 ──────────────────────────────────────────
    text_y = y + h - bar_h - 20
    # 卡片换行宽度：卡片宽约 646px，英文字体 10pt ≈ 6px/char，留边距后约 75 字符/行
    card_wrap_width = 70
    for i, item in enumerate(items[:8]):  # 最多显示 8 条
        wrapped = _wrap_text(item, width=card_wrap_width)
        for j, line in enumerate(wrapped[:4]):  # 每条最多 4 行
            if j == 0:
                prefix = "• "
            else:
                prefix = "  "
            ax.text(
                x + 12, text_y,
                prefix + line,
                ha="left", va="top",
                fontproperties=fonts["small"],
                fontsize=10,
                color=color_scheme["text"],
                zorder=4,
            )
            text_y -= 19
        text_y -= 8  # 条目间距


def _draw_timeline_section(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    items: list[str],
    bg_color: str,
    border_color: str,
    fonts: dict[str, FontProperties],
) -> None:
    """绘制时间轴上的一个区块（近期 / 远期）。

    Args:
        ax: matplotlib Axes 对象。
        x, y: 区块左下角坐标。
        w, h: 区块宽高。
        label: 区块标题（如"近期 2-5 年"）。
        items: 趋势项列表。
        bg_color: 背景色。
        border_color: 边框色。
        fonts: 字体字典。
    """
    # ── 背景 ──────────────────────────────────────────────
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=2,
        zorder=2,
    )
    ax.add_patch(rect)

    # ── 标题栏 ────────────────────────────────────────────
    header_h = 34
    ax.text(
        x + w / 2, y + h - header_h / 2 - 4,
        label,
        ha="center", va="center",
        fontproperties=fonts["subtitle"],
        fontsize=14,
        color=TEXT_DARK,
        zorder=4,
    )

    # 标题下分割线
    ax.plot(
        [x + 30, x + w - 30],
        [y + h - header_h - 10, y + h - header_h - 10],
        color=border_color, linewidth=1, alpha=0.5, zorder=4,
    )

    # ── 趋势列表 ──────────────────────────────────────────
    text_y = y + h - header_h - 30
    # 时间轴换行宽度：区块宽约 870px，英文字体 10.5pt ≈ 6.3px/char，留边距后约 80 字符/行
    timeline_wrap_width = 80
    for item in items[:10]:  # 最多 10 条
        wrapped = _wrap_text(item, width=timeline_wrap_width)
        for j, line in enumerate(wrapped[:4]):  # 每条最多 4 行
            prefix = "› " if j == 0 else "  "
            ax.text(
                x + 14, text_y,
                prefix + line,
                ha="left", va="top",
                fontproperties=fonts["small"],
                fontsize=10.5,
                color=TEXT_MEDIUM,
                zorder=4,
            )
            text_y -= 20
        text_y -= 7  # 条目间距


# ── 主函数 ──────────────────────────────────────────────────────────

def generate_png(data: dict[str, Any], output_path: str) -> None:
    """根据结构化数据生成信息图并保存为 PNG。

    图片布局：
    ┌──────────────────────────────────────┐
    │        大标题 + 副标题（core_goal）    │
    ├──────────┬──────────┬───────────────┤
    │ Hardware │Algorithm │ Application   │  ← 三张卡片
    │  卡片    │  卡片     │   卡片        │
    ├──────────┴──────────┴───────────────┤
    │  近期 2-5 年  ──→──  远期 6-10 年   │  ← 时间轴
    │  趋势列表         趋势列表          │
    └──────────────────────────────────────┘

    Args:
        data: LLM 提取结果字典，需包含:
              - three_layers: [{"layer": str, "key_characteristics": [str]}]
              - timeline: {"short_term_2_5_years": [str], "long_term_6_10_years": [str]}
              - core_goal: {"definition": str, "current_baseline": str, "target_metric": str}
        output_path: 输出 PNG 文件的路径。

    Raises:
        ImportError: 未安装 matplotlib 时。
    """
    fonts = _setup_fonts()

    # ── 创建画布 ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 16), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 2200)
    ax.set_ylim(0, 1600)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    # ── 提取数据 ──────────────────────────────────────────
    core_goal: dict[str, str] = data.get("core_goal", {})
    layers: list[dict[str, Any]] = data.get("three_layers", [])
    timeline: dict[str, list[str]] = data.get("timeline", {})

    definition = core_goal.get("definition", "")
    baseline = core_goal.get("current_baseline", "")
    target = core_goal.get("target_metric", "")

    short_term: list[str] = timeline.get("short_term_2_5_years", [])
    long_term: list[str] = timeline.get("long_term_6_10_years", [])

    # ── 1. 标题区域 ───────────────────────────────────────
    ax.text(
        1100, 1540,
        "AI+HW 2035: 1000× Intelligence per Joule",
        ha="center", va="center",
        fontproperties=fonts["title"],
        fontsize=32,
        color=TITLE_COLOR,
        zorder=5,
    )

    # 标题下方分割线
    divider_y = 1500
    ax.plot(
        [300, 1900], [divider_y, divider_y],
        color="#CCCCCC", linewidth=1.5, zorder=5,
    )

    # 核心目标定义（分割线下方）
    if definition:
        wrapped_def = _wrap_text(definition, width=80)
        sub_y = divider_y - 28
        for line in wrapped_def[:3]:
            ax.text(
                1100, sub_y,
                line,
                ha="center", va="center",
                fontproperties=fonts["subtitle"],
                fontsize=13,
                color=SUBTITLE_COLOR,
                zorder=5,
            )
            sub_y -= 20

    # ── 2. 三张卡片（横向填满，整体下移）───────────────────
    card_margin = 100         # 左右边距
    card_gap = 30             # 卡片间距
    total_card_w = 2200 - 2 * card_margin  # 可用总宽度
    card_w = (total_card_w - 2 * card_gap) // 3  # 单张卡片宽度
    card_h = 420
    card_y = 940              # 卡片 Y 坐标
    card_start_x = card_margin

    for i, layer_data in enumerate(layers):
        layer_name = layer_data.get("layer", f"Layer {i+1}")
        characteristics = layer_data.get("key_characteristics", [])

        # 匹配配色
        name_lower = layer_name.strip().lower()
        if "hardware" in name_lower:
            scheme = CARD_COLORS["Hardware"]
        elif "algorithm" in name_lower:
            scheme = CARD_COLORS["Algorithm"]
        elif "application" in name_lower:
            scheme = CARD_COLORS["Application"]
        else:
            # 默认按顺序分配颜色
            defaults = list(CARD_COLORS.values())
            scheme = defaults[i % len(defaults)]

        cx = card_start_x + i * (card_w + card_gap)
        _draw_card(
            ax, cx, card_y, card_w, card_h,
            title=layer_name,
            items=characteristics,
            color_scheme=scheme,
            fonts=fonts,
        )

    # ── 3. 时间轴（下方区域）─────────────────────────────────
    timeline_y = 50
    timeline_h = 800
    section_w = (2200 - 2 * card_margin - 260) // 2  # 左右各一块，中间留 260 给箭头
    section_h = 760
    arrow_w = 260

    left_x = card_margin
    right_x = left_x + section_w + arrow_w

    # 左侧：近期 2-5 年
    _draw_timeline_section(
        ax, left_x, timeline_y, section_w, section_h,
        label="近期 2—5 年",
        items=short_term,
        bg_color=TIMELINE_LEFT_COLOR,
        border_color=TIMELINE_LEFT_BORDER,
        fonts=fonts,
    )

    # 右侧：远期 6-10 年
    _draw_timeline_section(
        ax, right_x, timeline_y, section_w, section_h,
        label="远期 6—10 年",
        items=long_term,
        bg_color=TIMELINE_RIGHT_COLOR,
        border_color=TIMELINE_RIGHT_BORDER,
        fonts=fonts,
    )

    # 中间箭头
    arrow_cx = left_x + section_w + arrow_w / 2
    arrow_cy = timeline_y + section_h / 2
    ax.annotate(
        "", xy=(right_x - 12, arrow_cy),
        xytext=(left_x + section_w + 12, arrow_cy),
        arrowprops=dict(
            arrowstyle="->",
            lw=5,
            color=ARROW_COLOR,
            connectionstyle="arc3,rad=0",
        ),
        zorder=5,
    )
    # 箭头上方标注
    ax.text(
        arrow_cx, arrow_cy + 50,
        "技术演进",
        ha="center", va="center",
        fontproperties=fonts["small"],
        fontsize=13,
        color=ARROW_COLOR,
        zorder=5,
    )

    # ── 4. 底部水印 ────────────────────────────────────────
    ax.text(
        1100, 20,
        "Generated by AI+HW 2035 Analyzer",
        ha="center", va="center",
        fontproperties=fonts["small"],
        fontsize=9,
        color="#BBBBBB",
        zorder=5,
    )

    # ── 保存 ──────────────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(out_path),
        dpi=150,
        bbox_inches="tight",
        facecolor=BG_COLOR,
        edgecolor="none",
    )
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
