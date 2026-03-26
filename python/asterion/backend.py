from __future__ import annotations

import importlib.util
import os
import tempfile

from .cache import Cache
from .codegen import generate
from .environment import command
from .io import ValidationError
from .ir import canonicalize, verify, verify_schedule


def require_gpu():
    scheduler_guard = os.environ.get("ASTERION_REQUIRE_SLURM", "0")
    if scheduler_guard not in {"0", "1"}:
        raise ValidationError("ASTERION_REQUIRE_SLURM must be 0 or 1")
    if scheduler_guard == "1" and not os.environ.get("SLURM_JOB_ID"):
        raise ValidationError("cluster policy requires an authorized Slurm GPU allocation")
    try:
        import torch
        import triton
    except ImportError as exc:
        raise ValidationError(
            "install the optional GPU dependencies in the intended environment"
        ) from exc
    if not torch.cuda.is_available():
        raise ValidationError(
            "CUDA unavailable: use a compatible CUDA device; cluster runs require a Slurm GPU allocation"
        )
    return torch, triton


_TARGET_CACHE = {}


def target_manifest():
    torch, triton = require_gpu()
    device = torch.cuda.current_device()
    if device in _TARGET_CACHE:
        return dict(_TARGET_CACHE[device])
    props = torch.cuda.get_device_properties(device)
    result = {
        "name": props.name,
        "capability": list(torch.cuda.get_device_capability()),
        "total_memory": props.total_memory,
        "uuid": str(getattr(props, "uuid", "unavailable")),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_build": torch.version.cuda,
        "driver": command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "mig_inventory": command(["nvidia-smi", "-L"]),
    }
    capability = tuple(result["capability"])
    if triton.__version__ == "3.3.1":
        if capability < (7, 5):
            raise ValidationError("legacy backend requires compute capability 7.5 or newer")
    elif capability < (8, 0):
        raise ValidationError(
            "modern backend requires compute capability 8.0 or newer; SM75 needs the validated legacy stack"
        )
    _TARGET_CACHE[device] = result
    return dict(result)


class Kernel:
    def __init__(self, program, schedule):
        self.torch, self.triton = require_gpu()
        self.program = canonicalize(program)
        self.schedule = verify_schedule(self.program, schedule)
        source = generate(self.program, schedule)
        self.target = target_manifest()
        from .validation import kernel_key

        self.key = kernel_key(self.program, schedule, self.target)
        cache = Cache.default()
        payload = cache.get(self.key)
        if payload is None:
            cache.put(
                self.key,
                {
                    "source": source,
                    "program": self.program.to_dict(),
                    "schedule": schedule.to_dict(),
                    "target": self.target,
                },
            )
        elif payload["source"] != source:
            raise ValidationError("cache source does not match regenerated kernel")
        path = cache.root / (self.key + ".py")
        # Executed code is regenerated from verified IR, never arbitrary cached source.
        if not path.exists() or path.read_text() != source:
            fd, tmp = tempfile.mkstemp(dir=cache.root, suffix=".py")
            with os.fdopen(fd, "w") as out:
                out.write(source)
            os.replace(tmp, path)
        spec = importlib.util.spec_from_file_location("asterion_generated_" + self.key, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module
        self.source_path = str(path)

    def prepare(self, *inputs):
        torch = self.torch
        p = self.program
        shapes = (
            [(p.shape[0], p.shape[2]), (p.shape[2], p.shape[1])]
            if p.operation == "gemm"
            else [p.shape]
        )
        if len(inputs) != len(shapes):
            raise ValidationError("wrong number of input tensors")
        for x, shape in zip(inputs, shapes):
            if not x.is_cuda or x.device.index != torch.cuda.current_device():
                raise ValidationError("inputs must be on the current CUDA device")
            if x.dtype != torch.float32 or not x.is_contiguous() or tuple(x.shape) != shape:
                raise ValidationError("input shape/dtype/contiguity does not match verified IR")
            if x.requires_grad:
                raise ValidationError("Asterion v0.1 inference kernels do not implement autograd")
        out_shape = (
            (p.shape[0], p.shape[1])
            if p.operation == "gemm"
            else (p.shape[0],)
            if p.operation in {"sum", "max"}
            else p.shape
        )
        output = torch.empty(out_shape, device=inputs[0].device, dtype=torch.float32)
        stream = torch.cuda.current_stream()
        for x in (*inputs, output):
            x.record_stream(stream)
        buffers = {n.name: x for n, x in zip((n for n in p.nodes if n.op == "input"), inputs)}
        stages = []
        if p.operation == "pointwise" and not self.schedule.fusion:
            ranks = verify(p)
            for n in p.nodes:
                if n.op == "input" or ranks[n.name] == "scalar":
                    continue
                value = output if n.name == p.output else torch.empty_like(inputs[0])
                value.record_stream(stream)
                dependencies = list(dict.fromkeys(a for a in n.args if ranks[a] != "scalar"))
                stages.append(
                    (
                        getattr(self.module, "stage_" + n.name),
                        [buffers[a] for a in dependencies],
                        value,
                    )
                )
                buffers[n.name] = value
        return BoundKernel(self, inputs, output, stages)

    def __call__(self, *inputs):
        bound = self.prepare(*inputs)
        bound.launch()
        return bound.output


class BoundKernel:
    def __init__(self, kernel, inputs, output, stages):
        self.kernel, self.inputs, self.output, self.stages = kernel, inputs, output, stages
        self.compiled_kernels = {}
        self.input_shapes = tuple(tuple(x.shape) for x in inputs)
        self.output_shape = tuple(output.shape)
        self.storage_pointers = tuple(x.data_ptr() for x in (*inputs, output))

    def launch(self):
        k = self.kernel
        p, s = k.program, k.schedule
        for tensor, shape in zip(
            (*self.inputs, self.output), (*self.input_shapes, self.output_shape)
        ):
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != k.torch.float32
                or not tensor.is_contiguous()
            ):
                raise ValidationError("prepared tensor metadata changed")
            if tensor.device.index != k.torch.cuda.current_device():
                raise ValidationError("prepared tensors must remain on the current CUDA device")
        if any(x.requires_grad for x in (*self.inputs, self.output)):
            raise ValidationError("prepared inference tensors now require autograd")
        if tuple(x.data_ptr() for x in (*self.inputs, self.output)) != self.storage_pointers:
            raise ValidationError("prepared tensor storage changed")
        options = {"num_warps": s.warps, "enable_fp_fusion": False}
        if p.operation == "gemm":
            grid = (k.triton.cdiv(p.shape[0], s.tile_m), k.triton.cdiv(p.shape[1], s.tile_n))
            self.last_compiled = k.module.kernel[grid](
                *self.inputs, self.output, *p.shape, **options
            )
        elif p.operation == "pointwise" and not s.fusion and self.stages:
            grid = (k.triton.cdiv(p.shape[0], s.block),)
            for fn, args, out in self.stages:
                self.last_compiled = fn[grid](*args, out, p.shape[0], **options)
                self.compiled_kernels[getattr(self.last_compiled, "name", str(fn))] = (
                    self.last_compiled
                )
        else:
            grid = (k.triton.cdiv(p.shape[0], s.block if p.operation == "pointwise" else s.rows),)
            self.last_compiled = k.module.kernel[grid](
                *self.inputs, self.output, *p.shape, **options
            )
        self.compiled_kernels[getattr(self.last_compiled, "name", "kernel")] = self.last_compiled
        # Record current stream again when a prepared launcher is reused on another stream.
        stream = k.torch.cuda.current_stream()
        for x in (*self.inputs, self.output):
            x.record_stream(stream)
        for _, _, out in self.stages:
            out.record_stream(stream)
        return self.output


def reference(program, *inputs):
    import torch

    values = {}
    operands = iter(inputs)
    for n in program.nodes:
        if n.op == "input":
            v = next(operands)
        elif n.op == "const":
            v = float(n.value)
        elif n.op in {"sum", "max"}:
            v = values[n.args[0]].sum(dim=1) if n.op == "sum" else values[n.args[0]].amax(dim=1)
        elif n.op == "matmul":
            v = values[n.args[0]] @ values[n.args[1]]
        else:
            args = [values[a] for a in n.args]
            if program.operation not in {"pointwise", "gemm"}:
                args = [
                    x[:, None]
                    if hasattr(x, "ndim")
                    and x.ndim == 1
                    and any(hasattr(a, "ndim") and a.ndim == 2 for a in args)
                    else x
                    for x in args
                ]
            if n.op == "add":
                v = args[0] + args[1]
            elif n.op == "mul":
                v = args[0] * args[1]
            elif n.op == "sub":
                v = args[0] - args[1]
            elif n.op == "div":
                v = args[0] / args[1]
            elif n.op == "relu":
                v = torch.relu(
                    torch.as_tensor(args[0], device=inputs[0].device, dtype=inputs[0].dtype)
                )
            elif n.op == "exp":
                v = torch.exp(
                    torch.as_tensor(args[0], device=inputs[0].device, dtype=inputs[0].dtype)
                )
            else:
                raise ValidationError("unsupported reference node")
        values[n.name] = v
    return values[program.output]


def make_inputs(program, seed):
    torch, _ = require_gpu()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shapes = (
        [(program.shape[0], program.shape[2]), (program.shape[2], program.shape[1])]
        if program.operation == "gemm"
        else [program.shape]
    )
    return tuple(
        torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator)
        for shape in shapes
    )


def binary_metadata(bound):
    from hashlib import sha256

    total = 0
    hashes = []
    for compiled in bound.compiled_kernels.values():
        cubin = getattr(compiled, "asm", {}).get("cubin")
        if not isinstance(cubin, bytes) or not cubin:
            raise ValidationError("compiled CUBIN unavailable; cannot enforce machine-code budget")
        total += len(cubin)
        hashes.append(sha256(cubin).hexdigest())
    if not hashes:
        raise ValidationError("kernel must be launched before inspecting compiled code")
    return {
        "compiled_binary_hashes": sorted(hashes),
        "generated_code_bytes": total,
        "code_budget_unit": "cubin_bytes",
    }
