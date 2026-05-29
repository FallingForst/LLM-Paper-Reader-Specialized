"""
test.py — 信息图生成测试脚本

使用 output/extracted_data.json 中的结构化数据，调用 infographic.generate_png
生成测试图片，验证图片生成函数是否正常工作。

用法：
    python test.py
"""

import json
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from function.infographic import generate_png

# ── 路径常量 ──────────────────────────────────────────────────────
JSON_INPUT = _PROJECT_ROOT / "output" / "extracted_data.json"
PNG_OUTPUT = _PROJECT_ROOT / "output" / "test_briefing.png"


def main() -> None:
    """加载 JSON 数据并生成测试信息图。"""

    # 1. 检查输入文件
    if not JSON_INPUT.exists():
        print(f"❌ 输入文件不存在: {JSON_INPUT}")
        sys.exit(1)

    # 2. 加载 JSON
    print(f"📄 加载数据: {JSON_INPUT}")
    with open(JSON_INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 3. 打印摘要
    cg = data.get("core_goal", {})
    layers = data.get("three_layers", [])
    tl = data.get("timeline", {})

    print(f"   core_goal.definition: {cg.get('definition', '')[:60]}...")
    print(f"   three_layers: {len(layers)} 层")
    for layer in layers:
        name = layer.get("layer", "?")
        count = len(layer.get("key_characteristics", []))
        print(f"     - {name}: {count} 条特征")
    print(f"   timeline.short_term: {len(tl.get('short_term_2_5_years', []))} 条")
    print(f"   timeline.long_term:  {len(tl.get('long_term_6_10_years', []))} 条")

    # 4. 生成图片
    print(f"🖼️  生成信息图: {PNG_OUTPUT}")
    generate_png(data, str(PNG_OUTPUT))

    # 5. 验证输出
    if PNG_OUTPUT.exists():
        size_kb = PNG_OUTPUT.stat().st_size / 1024
        print(f"✅ 测试通过！图片已生成 ({size_kb:.1f} KB)")
    else:
        print("❌ 图片未生成")
        sys.exit(1)


if __name__ == "__main__":
    main()
