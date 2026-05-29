# function 包 —— 论文处理工具集
#
# 使用示例：
#   from function import transfer_to_text, load_and_chunk, extract_info_via_llm, generate_png
#   from function.llm_extract import process_paper

from .pdf_to_text import transfer_to_text
from .chunck import load_and_chunk
from .llm_extract import extract_info_via_llm, process_paper
from .infographic import generate_png

__all__ = [
    "transfer_to_text",
    "load_and_chunk",
    "extract_info_via_llm",
    "process_paper",
    "generate_png",
]
