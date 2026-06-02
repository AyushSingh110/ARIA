from __future__ import annotations

import ast
import math
import operator as op
from typing import Union

from langchain_core.tools import tool

# Safe operator whitelist — prevents arbitrary code execution
_SAFE_OPS: dict = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

_SAFE_NAMES: dict = {
    "pi": math.pi,
    "e": math.e,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
}


def _safe_eval(node: ast.AST) -> Union[int, float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _SAFE_NAMES:
        return _SAFE_NAMES[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id
        if func_name in _SAFE_NAMES and callable(_SAFE_NAMES[func_name]):
            args = [_safe_eval(a) for a in node.args]
            return _SAFE_NAMES[func_name](*args)
    raise ValueError(f"Unsafe expression component: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a safe mathematical expression.

    Supports: +, -, *, /, **, %, //, and functions like sqrt, log, sin, cos, tan, abs.
    Constants: pi, e.

    Args:
        expression: A mathematical expression string, e.g. "2 ** 10 + sqrt(144)"

    Returns:
        The numeric result as a string, or an error message.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except ValueError as exc:
        return f"Error: {exc}"
    except SyntaxError:
        return f"Error: invalid expression syntax — '{expression}'"
