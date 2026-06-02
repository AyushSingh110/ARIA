from aria.tools.calculator import calculator
from aria.tools.file_ops import write_file, read_file
from aria.tools.web_search import web_search

EXECUTOR_TOOLS = [calculator, web_search, write_file, read_file]

__all__ = ["calculator", "write_file", "read_file", "web_search", "EXECUTOR_TOOLS"]
