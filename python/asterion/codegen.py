"""Verified SSA and explicit schedules -> shape-polymorphic Triton source."""

from .ir import canonicalize, verify, verify_schedule

HEADER = "import triton\nimport triton.language as tl\n\n"


def expression(node, ranks, row_mode):
    args = []
    for a in node.args:
        broadcast = (
            row_mode
            and ranks[a] == "row"
            and ranks[node.name] == "tensor"
            and node.op not in {"sum", "max"}
        )
        args.append(a + "[:, None]" if broadcast else a)
    if node.op == "const":
        return repr(float(node.value))
    if node.op in {"add", "mul", "sub", "div"}:
        symbol = {"add": "+", "mul": "*", "sub": "-", "div": "/"}[node.op]
        return f"({args[0]} {symbol} {args[1]})"
    if node.op == "relu":
        return f"tl.maximum({args[0]}, 0.0)"
    if node.op == "exp":
        return f"tl.exp({args[0]})"
    if node.op in {"sum", "max"}:
        neutral = "0.0" if node.op == "sum" else '-float("inf")'
        return f"tl.{node.op}(tl.where(mask, {args[0]}, {neutral}), axis=1)"
    raise ValueError("unsupported expression node")


