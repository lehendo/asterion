"""Generate a standalone trusted script from verified IR, never imported source."""

from __future__ import annotations

import json
from pathlib import Path

from .codegen import generate
from .diagnosis import specification
from .io import ValidationError, atomic_json
from .ir import Program, Schedule

# This harness intentionally executes the two accompanying generated modules.
# Generation accepts only verified operation IR; opening a passive report does
# not run this script. Like all Python scripts, execute only trusted bundles.
HARNESS = '''"""Standalone FP32 regression witness. Run only trusted Python bundles.
Requires torch and Triton compatible with the recorded environment.
Device graph time and prepared completion wall time are different boundaries.
"""
import importlib.util
import json
import math
import random
import statistics
import time
from pathlib import Path
import torch
import triton

root = Path(__file__).resolve().parent
spec = json.loads((root / "reproducer.json").read_text())
p = spec["program"]
shape = p["shape"]
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
shapes = [(shape[0], shape[2]), (shape[2], shape[1])] if p["operation"] == "gemm" else [shape]
generator = torch.Generator(device="cuda").manual_seed(spec["seed"])
inputs = [torch.randn(s, device="cuda", dtype=torch.float32, generator=generator) for s in shapes]

def reference(values):
    env = {}
    operands = iter(values)
    for node in p["nodes"]:
        op, name = node["op"], node["name"]
        args = [env[a] for a in node["args"]]
        if op == "input":
            v = next(operands)
        elif op == "const":
            v = node["value"]
        elif op == "sum":
            v = args[0].sum(1)
        elif op == "max":
            v = args[0].amax(1)
        elif op == "matmul":
            v = args[0] @ args[1]
        else:
            if any(isinstance(a, torch.Tensor) and a.ndim == 2 for a in args):
                args = [a[:, None] if isinstance(a, torch.Tensor) and a.ndim == 1 else a for a in args]
            if op == "add": v = args[0]+args[1]
            elif op == "sub": v = args[0]-args[1]
            elif op == "mul": v = args[0]*args[1]
            elif op == "div": v = args[0]/args[1]
            elif op == "exp": v = torch.exp(torch.as_tensor(args[0], device=values[0].device, dtype=values[0].dtype))
            elif op == "relu": v = torch.relu(torch.as_tensor(args[0], device=values[0].device, dtype=values[0].dtype))
            else: raise ValueError("unsupported reference operation")
        env[name] = v
    return env[p["output"]].float()

def prepare(name, tensors):
    module_spec = importlib.util.spec_from_file_location("witness_"+name, root/(name+".py"))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    s = spec[name]
    out_shape = shape[:2] if p["operation"] == "gemm" else shape[:1] if p["operation"] in {"sum", "max"} else shape
    output = torch.empty(out_shape, device="cuda", dtype=torch.float32)
    options = {"num_warps": s["warps"], "enable_fp_fusion": False}
    stages = []
    if p["operation"] == "pointwise" and not s["fusion"]:
        buffers = {n["name"]: t for n,t in zip((n for n in p["nodes"] if n["op"] == "input"), tensors)}
        constants = set()
        for n in p["nodes"]:
            if n["op"] == "const" or (n["op"] != "input" and all(a in constants for a in n["args"])):
                constants.add(n["name"])
        for n in p["nodes"]:
            if n["op"] == "input" or n["name"] in constants: continue
            value = output if n["name"] == p["output"] else torch.empty_like(tensors[0])
            deps = list(dict.fromkeys(a for a in n["args"] if a not in constants))
            stages.append((getattr(module, "stage_"+n["name"]), [buffers[a] for a in deps], value))
            buffers[n["name"]] = value
    def launch():
        if p["operation"] == "gemm":
            module.kernel[(triton.cdiv(shape[0],s["tile_m"]),triton.cdiv(shape[1],s["tile_n"]))](*tensors,output,*shape,**options)
        elif stages:
            for fn,args,value in stages:
                fn[(triton.cdiv(shape[0],s["block"]),)](*args,value,shape[0],**options)
        else:
            grid = triton.cdiv(shape[0],s["block"] if p["operation"] == "pointwise" else s["rows"])
            module.kernel[(grid,)](*tensors,output,*shape,**options)
        return output
    return launch

functions, graphs = {}, {}
for name in ("baseline", "candidate"):
    fn = prepare(name, inputs)
    torch.testing.assert_close(fn(),reference([x.double() for x in inputs]),rtol=spec["rtol"],atol=spec["atol"])
    for _ in range(5): fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(spec["launches"]): fn()
    graphs[name], functions[name] = graph, fn
    graph.replay()
torch.cuda.synchronize()
rng = random.Random(spec["seed"])
blocks = []
for i in range(spec["rounds"]):
    order = ["baseline", "candidate"]
    rng.shuffle(order)
    measurements = {}
    for name in order:
        torch.cuda.synchronize()
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record(); graphs[name].replay(); end.record(); end.synchronize()
        device = start.elapsed_time(end)*1000/spec["launches"]
        before = time.perf_counter()
        functions[name](); torch.cuda.current_stream().synchronize()
        measurements[name] = {"device_graph_us":device,"prepared_completion_us":(time.perf_counter()-before)*1e6}
    blocks.append({"index":i,"order":order,"measurements":measurements})
ratios = [b["measurements"]["candidate"]["device_graph_us"]/b["measurements"]["baseline"]["device_graph_us"] for b in blocks]
wins = sum(r > 1+spec["threshold"] for r in ratios)
pvalue = sum(math.comb(len(ratios),i) for i in range(wins,len(ratios)+1))/2**len(ratios)
better = sum(r < 1/(1+spec["threshold"]) for r in ratios)
improvement_p = sum(math.comb(len(ratios),i) for i in range(better,len(ratios)+1))/2**len(ratios)
direction = "regression" if pvalue <= .005 else "improvement" if improvement_p <= .005 else "inconclusive"
expectation = json.loads((root/"expectation.json").read_text()) if (root/"expectation.json").exists() else {"verdict":"unmeasured"}
result = {"blocks":blocks,"median_paired_ratio":statistics.median(ratios),"regression_p":pvalue,"improvement_p":improvement_p,"verdict":direction,"regression_preserved":pvalue <= .005,"practical_threshold":spec["threshold"],"expected":expectation,"qualitative_preserved":direction == expectation["verdict"] if expectation["verdict"] in {"regression","improvement"} else None,"target":str(torch.cuda.get_device_properties(0))}
(root/"standalone-results.json").write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k != "blocks"},indent=2))
'''


def export_reproducer(value, output):
    spec = specification(value)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValidationError("standalone output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    program = Program.from_dict(spec["program"])
    for name in ("baseline", "candidate"):
        (output / (name + ".py")).write_text(generate(program, Schedule.from_dict(spec[name])))
    atomic_json(output / "reproducer.json", spec)
    (output / "reproduce.py").write_text(HARNESS)
    (output / "USAGE.txt").write_text(
        "Trusted standalone regression witness. Requires compatible torch/Triton and CUDA.\n"
        "Run: python reproduce.py\n"
        "Generated modules execute only when this script is explicitly run.\n"
        "Correctness checks cover seeded normal input; full diagnosis contains the broader numerical suite.\n"
        "This capture is not a minimized witness unless reduction.json records accepted reductions and confirmation.\n"
        "Prepared host timing excludes Asterion metadata checks and is not comparable to its checked prepared boundary.\n"
    )
    # JSON representation is deliberately separate from the executable harness.
    (output / "program.json").write_text(json.dumps(program.to_dict(), indent=2) + "\n")
    return output / "reproduce.py"
