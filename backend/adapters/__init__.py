"""ARIA trace adapters — convert external agent framework traces to ARIA DiagnoseRequest format.

Available adapters:
  from adapters.langgraph_adapter import langgraph_to_aria
  from adapters.openai_adapter    import openai_to_aria
"""
from adapters.langgraph_adapter import langgraph_to_aria
from adapters.openai_adapter    import openai_to_aria

__all__ = ["langgraph_to_aria", "openai_to_aria"]
