"""generate_briefing.py —— 论文处理流水线: PDF→文本→分块→LLM提取→信息图。

用法: 编辑 config/api-key.json 填入 API 密钥, 然后 python generate_briefing.py
"""

import json
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from function.pdf_to_text import transfer_to_text
from function.chunk import load_and_chunk
from function.llm_extract import extract_info_via_llm
from function.infographic import generate_png

INPUT_DIR = _PROJECT_ROOT / "input"
CONFIG_DIR = _PROJECT_ROOT / "config"
OUTPUT_DIR = _PROJECT_ROOT / "output"

INPUT_PDF = INPUT_DIR / "input.pdf"
CONFIG_JSON = CONFIG_DIR / "config.json"
PROMPT_JSON = CONFIG_DIR / "prompt.json"
API_KEY_JSON = CONFIG_DIR / "api-key.json"

TXT_OUTPUT = OUTPUT_DIR / "paper_text.txt"
JSON_OUTPUT = OUTPUT_DIR / "extracted_data.json"
PNG_OUTPUT = OUTPUT_DIR / "briefing.png"


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)-7s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("briefing")


def _load_api_key() -> str:
    """加载 API 密钥：config/api-key.json > 环境变量 DEEPSEEK_API_KEY。"""
    if API_KEY_JSON.exists():
        try:
            key_data = json.loads(API_KEY_JSON.read_text(encoding="utf-8"))
            key = (key_data.get("api_key", "") or "").strip()
            if key and not key.startswith("sk-xxx"):
                logger.debug("从 %s 加载 API 密钥", API_KEY_JSON)
                return key
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("无法解析 %s: %s", API_KEY_JSON, e)

    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if key:
        logger.debug("从环境变量 DEEPSEEK_API_KEY 加载 API 密钥")
        return key

    logger.error("未找到 API 密钥。请:\n  1) 编辑 %s 填入真实密钥, 或\n"
                 "  2) 设置环境变量: set DEEPSEEK_API_KEY=sk-xxxx", API_KEY_JSON)
    sys.exit(1)


# ── 主流程 ────────────────────────────────────────────────────────

def main() -> None:
    """按顺序执行 PDF → 文本 → 分块 → LLM 提取 → 信息图生成。"""
    logger.info("=" * 60)
    logger.info("AI+HW 2035 论文简报生成流水线 启动")
    logger.info("=" * 60)

    api_key = _load_api_key()
    if not INPUT_PDF.exists():
        logger.error("输入文件不存在: %s", INPUT_PDF)
        sys.exit(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 步骤 1: PDF → 文本
    logger.info("─" * 40)
    logger.info("步骤 1/4: PDF → 纯文本（自动过滤注释/参考文献）")
    try:
        transfer_to_text(str(INPUT_PDF), str(TXT_OUTPUT))
        output_text = TXT_OUTPUT.read_text(encoding="utf-8")
        para_count = len([p for p in output_text.split("\n\n") if p.strip()])
        logger.info("✓ PDF 转换完成 — %d 段落, %d 字符", para_count, len(output_text))
    except Exception as e:
        logger.exception("PDF 转换失败: %s", e)
        sys.exit(1)

    # 步骤 2: 分块
    logger.info("─" * 40)
    logger.info("步骤 2/4: 文本分块 (max=2000, overlap=200)")
    try:
        chunks = load_and_chunk(str(TXT_OUTPUT), max_chunk_size=2000,
                                chunk_overlap=200, preserve_sections=True)
        logger.info("✓ 分块完成，共 %d 个 chunk", len(chunks))
    except Exception as e:
        logger.exception("文本分块失败: %s", e)
        sys.exit(1)

    # 步骤 3: LLM 提取
    logger.info("─" * 40)
    logger.info("步骤 3/4: LLM 结构化信息提取 (%d chunks)", len(chunks))
    try:
        extracted = extract_info_via_llm(
            chunks=chunks, api_key=api_key,
            config_path=str(CONFIG_JSON), prompt_path=str(PROMPT_JSON))
        logger.info("✓ LLM 提取完成")
    except Exception as e:
        logger.exception("LLM 提取失败: %s", e)
        sys.exit(1)

    logger.info("  保存 JSON 备份: %s", JSON_OUTPUT)
    try:
        JSON_OUTPUT.write_text(json.dumps(extracted, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        logger.info("✓ JSON 备份已保存")
    except Exception as e:
        logger.warning("JSON 备份保存失败（不中断流程）: %s", e)

    # 步骤 4: 信息图
    logger.info("─" * 40)
    logger.info("步骤 4/4: 生成信息图 → %s", PNG_OUTPUT)
    try:
        generate_png(extracted, str(PNG_OUTPUT))
        logger.info("✓ 信息图生成完成")
    except Exception as e:
        logger.exception("信息图生成失败: %s", e)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("流水线全部完成！")
    logger.info("  文本: %s\n  JSON: %s\n  信息图: %s",
                TXT_OUTPUT, JSON_OUTPUT, PNG_OUTPUT)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
