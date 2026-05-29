# AI+HW 2035 论文简报生成流水线

基于 LLM 的学术论文自动化处理工具，将 PDF 格式的学术论文转换为结构化的 **PNG 信息图**（简报）。

> 当前面向论文：*《AI+HW 2035: Shaping the Next Decade》* —— 一篇关于 AI 硬件协同发展的愿景白皮书。

## 流水线概览

```
input.pdf  →  PDF 转文本  →  文本分块  →  LLM 结构化提取  →  PNG 信息图
```

| 步骤 | 模块 | 功能 |
|------|------|------|
| 1 | `pdf_to_text` | 使用 pdfplumber 提取 PDF 文本，智能识别段落边界 |
| 2 | `chunck` | 将纯文本按段落拆分为适合 LLM 处理的文本块（~2000 字符/块） |
| 3 | `llm_extract` | 调用 DeepSeek API 提取 core_goal、three_layers、timeline 结构化数据 |
| 4 | `infographic` | 使用 matplotlib 渲染包含三卡片 + 时间轴的 PNG 信息图 |

## 项目结构

```
.
├── generate_briefing.py      # 主控脚本（一键运行流水线）
├── requirements.txt          # Python 依赖清单
├── .gitignore                # Git 忽略规则
├── config/
│   ├── api-key.json          # API 密钥配置
│   ├── config.json           # LLM 参数与合并策略
│   └── prompt.json           # 系统提示词与用户模板
├── function/
│   ├── __init__.py           # 包初始化（统一导出）
│   ├── pdf_to_text.py        # PDF → 纯文本（智能分段：编号/缩进/标题识别）
│   ├── chunck.py             # 文本分块
│   ├── llm_extract.py        # LLM 结构化信息提取（含重试、合并、端到端接口）
│   └── infographic.py        # 信息图生成（matplotlib）
├── input/                    # 待处理的 PDF 文件（input.pdf）
└── output/                   # 输出目录
    ├── paper_text.txt        # 提取的纯文本
    ├── extracted_data.json   # LLM 提取的结构化数据
    └── briefing.png          # 最终信息图
```

## 快速开始

### 环境要求

- Python 3.9+
- 依赖库：`pdfplumber`、`openai`、`matplotlib`

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置 API 密钥

1. 编辑 `config/api-key.json`，填入你的 DeepSeek API 密钥：

```json
{
  "api_key": "sk-xxxxxxxxxxxxxxxx"
}
```

> 也可设置环境变量 `DEEPSEEK_API_KEY`，优先级低于 `api-key.json`。

### 运行

1. 将待处理的 PDF 文件放置到 `input/` 目录下，命名为 `input.pdf`
2. 运行主控脚本：

```bash
python generate_briefing.py
```

3. 查看 `output/` 目录获取结果：
   - `paper_text.txt` — 提取的纯文本
   - `extracted_data.json` — LLM 提取的结构化 JSON
   - `briefing.png` — 最终信息图

## 配置说明

### `config/config.json` — LLM 参数与合并策略

```json
{
  "llm": {
    "model": "deepseek-v4-pro",
    "base_url": "https://api.deepseek.com",
    "temperature": 0.1,
    "max_tokens": 4000,
    "max_retries": 2
  },
  "merge_strategy": {
    "core_goal": { "method": "deep_merge_first_non_empty" },
    "three_layers": { "method": "deduplicate_by_layer_merge_characteristics" },
    "timeline": { "method": "merge_arrays_deduplicate" }
  }
}
```

- **llm**：控制模型调用参数（温度、最大 Token 数、重试策略等）
- **merge_strategy**：多个文本块的 LLM 结果如何合并为最终 JSON

### `config/prompt.json` — 提示词配置

定义 LLM 的角色、提取指南和输出 JSON 结构：

| 提取字段 | 说明 |
|----------|------|
| `core_goal` | 核心目标（定义 / 当前基线 / 目标指标） |
| `three_layers` | 三个抽象层次（Hardware / Algorithm / Application）的技术特征 |
| `timeline` | 近期（2-5 年）与远期（6-10 年）技术趋势 |

> 提示词针对 "AI+HW 2035" 论文做了专门优化，若处理其他论文需修改 `prompt.json`。

## 输出示例

生成的信息图布局：

```
┌──────────────────────────────────────────────────┐
│     AI+HW 2035: 1000× Intelligence per Joule     │
│  ──────────────────────────────────────────────  │
│          core_goal 定义（分割线下方）              │
├────────────────┬────────────────┬───────────────┤
│   Hardware     │   Algorithm    │  Application  │  ← 3 卡片横向填满（各 6 条特征）
├────────────────┴────────────────┴───────────────┤
│   近期 2-5 年   ────→────  远期 6-10 年          │  ← 各 8 条趋势
└──────────────────────────────────────────────────┘
```

## 许可

本项目基于 [GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.html) 许可。
