"""ncnn built-in layer interface extraction.

Source-of-truth = ncnn/src/layer/<op>.h + <op>.cpp. Use parser.parse_layer()
for a single op (returns a dict) or extract_layer_interfaces.main() to dump
the whole 111-layer dictionary.
"""

from .parser import parse_layer, ParseResult
from .md_doc_loader import load_doc_table
from .lookup import (
    load_dict, get_interface, render_for_prompt,
    guess_layer_from_task, derive_params_from_dict,
)

__all__ = [
    "parse_layer", "ParseResult", "load_doc_table",
    "load_dict", "get_interface", "render_for_prompt",
    "guess_layer_from_task", "derive_params_from_dict",
]
