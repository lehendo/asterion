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


def generate(program, schedule):
    p = canonicalize(program)
    ranks = verify(p)
    verify_schedule(p, schedule)
    if p.operation == "gemm":
        s = schedule
        lines = [
            '@triton.jit(do_not_specialize=["M", "N", "K"])',
            "def kernel(X, Y, O, M, N, K):",
            f"    rm = tl.program_id(0)*{s.tile_m} + tl.arange(0, {s.tile_m})",
            f"    rn = tl.program_id(1)*{s.tile_n} + tl.arange(0, {s.tile_n})",
            f"    rk = tl.arange(0, {s.tile_k})",
            f"    acc = tl.full(({s.tile_m}, {s.tile_n}), 0, tl.float32)",
            f"    for outer in range(0, tl.cdiv(K, {s.tile_k * s.unroll})):",
            f"        for inner in tl.static_range(0, {s.unroll}):",
            f"            kk = (outer*{s.unroll}+inner)*{s.tile_k} + rk",
            "            a = tl.load(X + rm[:, None]*K + kk[None, :], (rm[:, None] < M) & (kk[None, :] < K), other=0)",
            "            b = tl.load(Y + kk[:, None]*N + rn[None, :], (kk[:, None] < K) & (rn[None, :] < N), other=0)",
            '            acc = tl.dot(a, b, acc, input_precision="ieee")',
            "    tl.store(O + rm[:, None]*N + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N))",
        ]
        return HEADER + "\n".join(lines) + "\n"
    if p.operation == "pointwise" and not schedule.fusion and len(p.nodes) > 1:
        chunks = []
        for node in p.nodes:
            if node.op == "input" or ranks[node.name] == "scalar":
                continue
            dependencies = list(dict.fromkeys(a for a in node.args if ranks[a] != "scalar"))
            lines = [
                '@triton.jit(do_not_specialize=["N"])',
                f"def stage_{node.name}(" + ", ".join("P_" + a for a in dependencies) + ", O, N):",
                f"    idx = tl.program_id(0)*{schedule.block} + tl.arange(0, {schedule.block})",
            ]
            for n in p.nodes:
                # Pure scalar SSA is hoisted into each materialized tensor stage.
                # It has no memory effects and never needs a tensor allocation.
                if ranks[n.name] == "scalar":
                    lines.append(f"    {n.name} = {expression(n, ranks, False)}")
            for a in dependencies:
                lines.append(f"    {a} = tl.load(P_{a} + idx, idx < N, other=0)")
            lines += [
                f"    {node.name} = {expression(node, ranks, False)}",
                f"    tl.store(O + idx, {node.name}, idx < N)",
            ]
            chunks.append("\n".join(lines))
        return HEADER + "\n\n".join(chunks) + "\n"
    row_mode = p.operation != "pointwise"
    if row_mode:
        block = schedule.row_tile or 1 << (p.shape[1] - 1).bit_length()
        lines = [
            '@triton.jit(do_not_specialize=["R", "C"])',
            "def kernel(X, O, R, C):",
            f"    rr = tl.program_id(0)*{schedule.rows} + tl.arange(0, {schedule.rows})",
            f"    cc = tl.arange(0, {block})",
            "    mask = (rr[:, None] < R) & (cc[None, :] < C)",
        ]
        other = '-float("inf")' if p.operation in {"softmax", "max"} else "0.0"
        input_expr = f"tl.load(X + rr[:, None]*C + cc[None, :], mask, other={other})"
    else:
        lines = [
            '@triton.jit(do_not_specialize=["N"])',
            "def kernel(X, O, N):",
            f"    idx = tl.program_id(0)*{schedule.block} + tl.arange(0, {schedule.block})",
        ]
        input_expr = "tl.load(X + idx, idx < N, other=0)"
    for node in p.nodes:
        value = input_expr if node.op == "input" else expression(node, ranks, row_mode)
        lines.append(f"    {node.name} = {value}")
    if not row_mode:
        lines += [f"    tl.store(O + idx, {p.output}, idx < N)"]
    elif ranks[p.output] == "row":
        lines += [f"    tl.store(O + rr, {p.output}, rr < R)"]
    else:
        lines += [f"    tl.store(O + rr[:, None]*C + cc[None, :], {p.output}, mask)"]
    return HEADER + "\n".join(lines) + "\n"
