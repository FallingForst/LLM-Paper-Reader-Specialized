"""LLM 信息提取工具 —— 使用 DeepSeek/OpenAI 兼容接口提取论文结构化信息。"""

import json
import time
from pathlib import Path
from typing import Any

_DEFAULT_DIR = Path(__file__).parent
_CONFIG_DIR = _DEFAULT_DIR.parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.json"
_DEFAULT_PROMPT_PATH = _CONFIG_DIR / "prompt.json"


def _load_json(path: Path) -> dict[str, Any]:
    """加载 JSON 文件。"""
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    return _load_json(path)


def _load_prompts(prompt_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(prompt_path) if prompt_path else _DEFAULT_PROMPT_PATH
    return _load_json(path)


def _resolve_prompt(prompts_cfg: dict[str, Any], key: str,
                    default: str = "") -> str:
    """解析提示词字段，兼容 str 与 list[str] 两种格式。"""
    value = prompts_cfg.get(key, default)
    if isinstance(value, list):
        return "\n".join(value)
    return value if isinstance(value, str) else default


# ── 合并策略 ────────────────────────────────────────────────────────

def _merge_results(results: list[dict[str, Any]],
                   merge_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """合并多个 chunk 的 LLM 提取结果。

    规则: core_goal 逐字段首次非空合并; three_layers 按 layer 去重合并 characteristics;
    timeline 各自数组去重合并。
    """
    if merge_config is None:
        merge_config = {}

    cg_fields = (merge_config.get("core_goal", {}).get("sub_fields", [])
                 or ["definition", "current_baseline", "target_metric"])
    tl_fields = (merge_config.get("timeline", {}).get("sub_fields", [])
                 or ["short_term_2_5_years", "long_term_6_10_years"])

    merged: dict[str, Any] = {
        "core_goal": {f: "" for f in cg_fields},
        "three_layers": [],
        "timeline": {f: [] for f in tl_fields},
    }

    seen_layers: set[str] = set()
    layer_chars: dict[str, set[str]] = {}

    for result in results:
        cg = result.get("core_goal")
        if isinstance(cg, dict):
            for field in cg_fields:
                val = (cg.get(field, "") or "").strip()
                if val and not merged["core_goal"][field]:
                    merged["core_goal"][field] = val

        layers = result.get("three_layers")
        if isinstance(layers, list):
            dk = merge_config.get("three_layers", {}).get("dedup_key", "layer")
            mf = merge_config.get("three_layers", {}).get("merge_field", "key_characteristics")
            for item in layers:
                if not isinstance(item, dict):
                    continue
                name = (item.get(dk, "") or "").strip()
                if not name:
                    continue
                chars = item.get(mf, []) or []
                if name not in seen_layers:
                    seen_layers.add(name)
                    layer_chars[name] = set()
                    merged["three_layers"].append({dk: name, mf: []})
                for ch in chars:
                    ch = ch.strip()
                    if ch and ch not in layer_chars[name]:
                        layer_chars[name].add(ch)
                        for entry in merged["three_layers"]:
                            if entry[dk] == name:
                                entry[mf].append(ch)
                                break

        tl = result.get("timeline")
        if isinstance(tl, dict):
            for field in tl_fields:
                items = tl.get(field, []) or []
                if not isinstance(items, list):
                    continue
                existing = set(merged["timeline"][field])
                for item in items:
                    item = item.strip()
                    if item and item not in existing:
                        existing.add(item)
                        merged["timeline"][field].append(item)
    return merged


# ── 主函数 ──────────────────────────────────────────────────────────

def extract_info_via_llm(
    chunks: list[str], api_key: str,
    config_path: str | Path | None = None,
    prompt_path: str | Path | None = None,
) -> dict[str, Any]:
    """对每个文本块调用 LLM 提取结构化信息并合并返回。"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("需要安装 openai 库。请执行: pip install openai")

    cfg = _load_config(config_path)
    prompts_cfg = _load_prompts(prompt_path)
    llm_cfg = cfg.get("llm", {})
    merge_cfg = cfg.get("merge_strategy", {})

    model = llm_cfg.get("model", "deepseek-v4-pro")
    base_url = llm_cfg.get("base_url", "https://api.deepseek.com")
    temperature = llm_cfg.get("temperature", 0.1)
    max_tokens = llm_cfg.get("max_tokens", 2000)
    max_retries = llm_cfg.get("max_retries", 2)
    retry_delays = llm_cfg.get("retry_delays_seconds", [1, 2])
    reasoning_effort = llm_cfg.get("reasoning_effort", "high")
    extra_body = llm_cfg.get("extra_body", {"thinking": {"type": "enabled"}})

    system_prompt = _resolve_prompt(prompts_cfg, "system", "")
    user_template = _resolve_prompt(prompts_cfg, "user_template", "{chunk_text}")

    client = OpenAI(api_key=api_key, base_url=base_url)
    all_results: list[dict[str, Any]] = []
    total = len(chunks)

    for idx, chunk in enumerate(chunks, 1):
        print(f"[{idx}/{total}] 正在处理 chunk (长度: {len(chunk)} 字符)...")
        result = _call_llm_with_retry(
            client=client, model=model, system_prompt=system_prompt,
            user_template=user_template, chunk_text=chunk,
            temperature=temperature, max_tokens=max_tokens,
            max_retries=max_retries, retry_delays=retry_delays,
            reasoning_effort=reasoning_effort, extra_body=extra_body,
        )
        if result is not None:
            all_results.append(result)
            cg = result.get("core_goal", {})
            preview = (cg.get("definition", "") if isinstance(cg, dict)
                       else str(cg))[:60]
            print(f"  ✓ core_goal: {preview}...")
        else:
            print(f"  ✗ 处理失败，已跳过该 chunk")

    if not all_results:
        raise RuntimeError("所有 chunk 的 LLM 调用均失败，无法生成结果")

    merged = _merge_results(all_results, merge_cfg)
    print(f"\n合并完成: {len(all_results)}/{total} 个 chunk 成功处理")
    return merged


# ── 带重试的 LLM 调用 ────────────────────────────────────────────────

def _call_llm_with_retry(
    client: Any, model: str, system_prompt: str, user_template: str,
    chunk_text: str, temperature: float = 0.1, max_tokens: int = 2000,
    max_retries: int = 2, retry_delays: list[int] | None = None,
    reasoning_effort: str = "high",
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """对单个 chunk 调用 LLM，失败自动重试。返回解析后的 dict 或 None。"""
    if retry_delays is None:
        retry_delays = [1, 2]
    if extra_body is None:
        extra_body = {"thinking": {"type": "enabled"}}

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_template.format(chunk_text=chunk_text)},
                ],
                temperature=temperature, max_tokens=max_tokens, stream=False,
                reasoning_effort=reasoning_effort, extra_body=extra_body,
            )
            raw = response.choices[0].message.content
            if raw is None:
                raise ValueError("LLM 返回了空响应")
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

def process_paper(pdf_path: str, api_key: str,
                  txt_path: str | None = None,
                  max_chunk_size: int = 2000, chunk_overlap: int = 0,
                  min_chunk_size: int = 0, min_paragraph_size: int = 50,
                  preserve_sections: bool = True,
                  config_path: str | Path | None = None,
                  prompt_path: str | Path | None = None) -> dict[str, Any]:
    """端到端处理：PDF → 文本 → 分块 → LLM 提取。"""
    from .pdf_to_text import transfer_to_text
    from .chunk import load_and_chunk

    if txt_path is None:
        txt_path = str(Path(pdf_path).with_suffix(".txt"))
    transfer_to_text(pdf_path, txt_path)
    print(f"PDF 已转换为文本: {txt_path}")

    chunks = load_and_chunk(txt_path, max_chunk_size=max_chunk_size,
                            chunk_overlap=chunk_overlap,
                            min_chunk_size=min_chunk_size,
                            min_paragraph_size=min_paragraph_size,
                            preserve_sections=preserve_sections)
    print(f"文本已分为 {len(chunks)} 个 chunk")
    return extract_info_via_llm(chunks, api_key, config_path=config_path,
                                prompt_path=prompt_path)


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
