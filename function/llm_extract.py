"""
LLM 信息提取工具

使用 DeepSeek / OpenAI 兼容接口，对文本块依次调用 LLM 提取学术论文的
结构化信息（核心目标、三层架构、时间线），并合并为统一结果。

配置文件：
- config.json  → LLM 调用参数、合并策略
- prompt.json  → 系统提示词、用户消息模板
"""

import json
import time
from pathlib import Path
from typing import Any


# ── 配置加载 ────────────────────────────────────────────────────────

# 默认配置文件路径（config/ 与本模块的 function/ 平级）
_DEFAULT_DIR = Path(__file__).parent  # function/
_CONFIG_DIR = _DEFAULT_DIR.parent / "config"  # d:\PROJ\AI\config\
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.json"
_DEFAULT_PROMPT_PATH = _CONFIG_DIR / "prompt.json"


def _load_json(path: Path) -> dict[str, Any]:
    """加载 JSON 文件，不存在时抛出 FileNotFoundError。"""
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """加载 LLM 调用参数与合并策略配置文件。"""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    return _load_json(path)


def _load_prompts(prompt_path: str | Path | None = None) -> dict[str, Any]:
    """加载提示词配置文件。"""
    path = Path(prompt_path) if prompt_path else _DEFAULT_PROMPT_PATH
    return _load_json(path)


def _resolve_prompt(
    prompts_cfg: dict[str, Any],
    key: str,
    default: str = "",
) -> str:
    """从提示词配置中解析字段值，兼容字符串与字符串数组两种格式。

    若值为 list[str]，则以换行符拼接；若为 str，直接返回。
    """
    value = prompts_cfg.get(key, default)
    if isinstance(value, list):
        return "\n".join(value)
    if isinstance(value, str):
        return value
    return default


# ── 合并策略 ────────────────────────────────────────────────────────

def _merge_results(
    results: list[dict[str, Any]],
    merge_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """将多个 chunk 的 LLM 提取结果合并为最终字典。

    合并规则由 config.json 中的 merge_strategy 定义：

    - core_goal（对象）：每个子字段（definition / current_baseline / target_metric）
      以最先出现的非空值为准（deep_merge_first_non_empty）。

    - three_layers（对象数组）：按 layer 字段去重，同一 layer 的
      key_characteristics 合并并去重（deduplicate_by_layer_merge_characteristics）。

    - timeline（对象）：合并 short_term_2_5_years 和 long_term_6_10_years
      各自的数组并去重（merge_arrays_deduplicate）。
    """
    if merge_config is None:
        merge_config = {}

    # ── 初始化默认结构 ──────────────────────────────────────
    core_goal_fields = (
        merge_config.get("core_goal", {}).get("sub_fields", [])
        or ["definition", "current_baseline", "target_metric"]
    )
    timeline_fields = (
        merge_config.get("timeline", {}).get("sub_fields", [])
        or ["short_term_2_5_years", "long_term_6_10_years"]
    )

    merged: dict[str, Any] = {
        "core_goal": {field: "" for field in core_goal_fields},
        "three_layers": [],
        "timeline": {field: [] for field in timeline_fields},
    }

    seen_layers: set[str] = set()
    # 为每个 layer 维护已见过的 key_characteristics
    layer_characteristics: dict[str, set[str]] = {}

    for result in results:
        # ── core_goal: 逐字段首次非空合并 ──────────────────
        cg = result.get("core_goal")
        if isinstance(cg, dict):
            for field in core_goal_fields:
                val = (cg.get(field, "") or "").strip()
                if val and not merged["core_goal"][field]:
                    merged["core_goal"][field] = val

        # ── three_layers: 按 layer 去重 + 合并 characteristics ──
        layers = result.get("three_layers")
        if isinstance(layers, list):
            dedup_key = (
                merge_config.get("three_layers", {}).get("dedup_key", "layer")
            )
            merge_field = (
                merge_config.get("three_layers", {}).get(
                    "merge_field", "key_characteristics"
                )
            )
            for item in layers:
                if not isinstance(item, dict):
                    continue
                layer_name = (item.get(dedup_key, "") or "").strip()
                if not layer_name:
                    continue
                chars = item.get(merge_field, []) or []

                if layer_name not in seen_layers:
                    seen_layers.add(layer_name)
                    layer_characteristics[layer_name] = set()
                    merged["three_layers"].append({
                        dedup_key: layer_name,
                        merge_field: [],
                    })

                # 合并 characteristics 并去重
                for ch in chars:
                    ch_clean = ch.strip()
                    if ch_clean and ch_clean not in layer_characteristics[layer_name]:
                        layer_characteristics[layer_name].add(ch_clean)
                        # 找到对应的 layer 条目追加
                        for entry in merged["three_layers"]:
                            if entry[dedup_key] == layer_name:
                                entry[merge_field].append(ch_clean)
                                break

        # ── timeline: 合并各数组并去重 ──────────────────────
        tl = result.get("timeline")
        if isinstance(tl, dict):
            for field in timeline_fields:
                items = tl.get(field, []) or []
                if not isinstance(items, list):
                    continue
                existing = set(merged["timeline"][field])
                for item in items:
                    item_clean = item.strip()
                    if item_clean and item_clean not in existing:
                        existing.add(item_clean)
                        merged["timeline"][field].append(item_clean)

    return merged


# ── 主函数 ──────────────────────────────────────────────────────────

def extract_info_via_llm(
    chunks: list[str],
    api_key: str,
    config_path: str | Path | None = None,
    prompt_path: str | Path | None = None,
) -> dict[str, Any]:
    """对每个文本块调用 LLM 提取结构化信息，并合并返回。

    提示词从 prompt.json 读取，模型参数与合并策略从 config.json 读取。

    Args:
        chunks: 文本块列表（通常来自 load_and_chunk 的输出）。
        api_key: API 密钥。
        config_path: LLM 参数配置文件路径，为 None 时使用默认 config.json。
        prompt_path: 提示词配置文件路径，为 None 时使用默认 prompt.json。

    Returns:
        合并后的字典，包含 core_goal、three_layers、timeline 三个字段。

    Raises:
        ImportError: 当未安装 openai 库时。
        FileNotFoundError: 当配置文件不存在时。
        RuntimeError: 当所有 chunk 均处理失败时。
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("需要安装 openai 库。请执行: pip install openai")

    # ── 加载配置 ──────────────────────────────────────────────
    cfg = _load_config(config_path)
    prompts_cfg = _load_prompts(prompt_path)

    llm_cfg: dict[str, Any] = cfg.get("llm", {})
    merge_cfg: dict[str, Any] = cfg.get("merge_strategy", {})

    model: str = llm_cfg.get("model", "deepseek-v4-pro")
    base_url: str = llm_cfg.get("base_url", "https://api.deepseek.com")
    temperature: float = llm_cfg.get("temperature", 0.1)
    max_tokens: int = llm_cfg.get("max_tokens", 2000)
    max_retries: int = llm_cfg.get("max_retries", 2)
    retry_delays: list[int] = llm_cfg.get("retry_delays_seconds", [1, 2])
    reasoning_effort: str = llm_cfg.get("reasoning_effort", "high")
    extra_body: dict[str, Any] = llm_cfg.get("extra_body", {"thinking": {"type": "enabled"}})

    system_prompt: str = _resolve_prompt(prompts_cfg, "system", "")
    user_template: str = _resolve_prompt(prompts_cfg, "user_template", "{chunk_text}")

    # ── 初始化客户端 ──────────────────────────────────────────
    client = OpenAI(api_key=api_key, base_url=base_url)

    all_results: list[dict[str, Any]] = []
    total = len(chunks)

    for idx, chunk in enumerate(chunks, 1):
        print(f"[{idx}/{total}] 正在处理 chunk (长度: {len(chunk)} 字符)...")

        result = _call_llm_with_retry(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_template=user_template,
            chunk_text=chunk,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_delays=retry_delays,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
        )

        if result is not None:
            all_results.append(result)
            cg = result.get("core_goal", {})
            if isinstance(cg, dict):
                preview = cg.get("definition", "")[:60]
            else:
                preview = str(cg)[:60]
            print(f"  ✓ core_goal: {preview}...")
        else:
            print(f"  ✗ 处理失败，已跳过该 chunk")

    if not all_results:
        raise RuntimeError("所有 chunk 的 LLM 调用均失败，无法生成结果")

    # ── 合并结果 ──────────────────────────────────────────────
    merged = _merge_results(all_results, merge_cfg)
    print(f"\n合并完成: {len(all_results)}/{total} 个 chunk 成功处理")
    return merged


# ── 带重试的 LLM 调用 ────────────────────────────────────────────────

def _call_llm_with_retry(
    client: Any,
    model: str,
    system_prompt: str,
    user_template: str,
    chunk_text: str,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    max_retries: int = 2,
    retry_delays: list[int] | None = None,
    reasoning_effort: str = "high",
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """对单个 chunk 调用 LLM，失败时自动重试。

    Args:
        client: OpenAI 客户端实例。
        model: 模型名称。
        system_prompt: 系统提示词。
        user_template: 用户消息模板（含 {chunk_text} 占位符）。
        chunk_text: 当前文本块内容。
        temperature: 采样温度。
        max_tokens: 最大生成 token 数。
        max_retries: 最大重试次数。
        retry_delays: 每次重试前的等待秒数列表。
        reasoning_effort: DeepSeek 推理强度（"high" / "medium" / "low"）。
        extra_body: 额外请求体参数（如启用 thinking 模式）。

    Returns:
        解析后的字典，或 None（全部重试失败时）。
    """
    if retry_delays is None:
        retry_delays = [1, 2]
    if extra_body is None:
        extra_body = {"thinking": {"type": "enabled"}}

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_template.format(
                            chunk_text=chunk_text
                        ),
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                reasoning_effort=reasoning_effort,
                extra_body=extra_body,
            )

            raw = response.choices[0].message.content
            if raw is None:
                raise ValueError("LLM 返回了空响应")

            # 尝试提取 JSON（处理可能的 markdown 代码块包裹）
            raw = raw.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1]) if len(lines) >= 3 else raw

            parsed = json.loads(raw)

            if not isinstance(parsed, dict):
                raise ValueError("LLM 返回的不是 JSON 对象")

            return parsed

        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            print(f"  解析失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries and attempt < len(retry_delays):
                time.sleep(retry_delays[attempt])

        except Exception as e:
            last_error = e
            print(f"  调用失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries and attempt < len(retry_delays):
                time.sleep(retry_delays[attempt] * 2)

    print(f"  已达最大重试次数，最终错误: {last_error}")
    return None


# ── 便捷入口：端到端处理 ────────────────────────────────────────────

def process_paper(
    pdf_path: str,
    api_key: str,
    txt_path: str | None = None,
    max_chunk_size: int = 2000,
    chunk_overlap: int = 0,
    min_chunk_size: int = 0,
    min_paragraph_size: int = 50,
    preserve_sections: bool = True,
    config_path: str | Path | None = None,
    prompt_path: str | Path | None = None,
) -> dict[str, Any]:
    """端到端处理：PDF → 文本 → 分块 → LLM 提取。

    Args:
        pdf_path: 输入的 PDF 论文路径。
        api_key: API 密钥。
        txt_path: 中间文本文件保存路径（可选，默认自动生成）。
        max_chunk_size: 分块最大字符数，默认 2000。
        chunk_overlap: 相邻 chunk 重叠字符数，0=不重叠。推荐 100~200。
        min_chunk_size: 最小 chunk 大小，过短的末尾 chunk 会被合并。
        min_paragraph_size: 短段落合并阈值（字符数），默认 50。
        preserve_sections: 是否在节标题处优先切分，默认 True。
        config_path: LLM 参数配置文件路径，为 None 时使用默认 config.json。
        prompt_path: 提示词配置文件路径，为 None 时使用默认 prompt.json。

    Returns:
        合并后的结构化信息字典。
    """
    from .pdf_to_text import transfer_to_text
    from .chunk import load_and_chunk

    # 1. PDF → 文本
    if txt_path is None:
        txt_path = str(Path(pdf_path).with_suffix(".txt"))
    transfer_to_text(pdf_path, txt_path)
    print(f"PDF 已转换为文本: {txt_path}")

    # 2. 文本 → 分块
    chunks = load_and_chunk(
        txt_path,
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        min_paragraph_size=min_paragraph_size,
        preserve_sections=preserve_sections,
    )
    print(f"文本已分为 {len(chunks)} 个 chunk")

    # 3. LLM 提取（参数来自 config.json，提示词来自 prompt.json）
    return extract_info_via_llm(
        chunks, api_key, config_path=config_path, prompt_path=prompt_path
    )


# ── 自测（需提供 API key）───────────────────────────────────────────
if __name__ == "__main__":
    import os

    # 优先从 api-key.json 读取，回退到环境变量
    api_key = ""
    key_file = _DEFAULT_DIR.parent / "config" / "api-key.json"
    if key_file.exists():
        try:
            with open(key_file, "r", encoding="utf-8") as f:
                key_data = json.load(f)
            api_key = (key_data.get("api_key", "") or "").strip()
            if api_key.startswith("sk-xxx"):
                api_key = ""
        except (json.JSONDecodeError, OSError):
            pass
    if not api_key:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("请设置 API 密钥:\n  1) 编辑 config/api-key.json, 或\n  2) 设置环境变量 DEEPSEEK_API_KEY")
        exit(1)

    # 示例：使用已有 chunks 测试
    sample_chunks = [
        "本文提出了一种基于深度学习的文本分类方法。该方法分为三个层次："
        "词嵌入层、序列编码层和注意力聚合层。2020年我们完成了初步实验，"
        "2021年在公开数据集上验证了效果。",
        "实验结果表明，该方法在 SST-2 数据集上达到了 94.3% 的准确率。"
        "未来工作包括将该方法扩展到多语言场景。",
    ]

    result = extract_info_via_llm(sample_chunks, api_key)
    print("\n========== 最终结果 ==========")
    print(json.dumps(result, ensure_ascii=False, indent=2))
