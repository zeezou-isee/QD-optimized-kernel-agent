"""Proposer (Workflow §3): LLM turns the baseline kernel into parameterized templates."""

from .proposer import LLMProposer, parse_template
from .prompts import proposer_prompt

__all__ = ["LLMProposer", "parse_template", "proposer_prompt"]
