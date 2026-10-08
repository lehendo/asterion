"""Run a certified portfolio on the current server GPU, including stream dependencies."""

from __future__ import annotations

import argparse
import json

from asterion.backend import make_inputs, require_gpu
from asterion.gpu import operation_reference
from asterion.io import read
from asterion.ir import make_program
from asterion.runtime import Runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("portfolio")
    parser.add_argument("--backend", choices=["triton", "native"], default="triton")
    parser.add_argument("--shape", type=int, nargs="+")
    parser.add_argument("--replays", type=int, default=3)
    args = parser.parse_args()
    if args.replays < 1:
        parser.error("--replays must be positive")
    portfolio = read(args.portfolio)
    certificates = portfolio.get("validation_certificates", [])
    shape = args.shape or (certificates[0]["shape"] if certificates else None)
    if shape is None:
        parser.error("supply --shape or a portfolio with a numerical certificate")
    torch, _ = require_gpu()
    inputs = make_inputs(make_program(portfolio["operation"], shape), 1729)
    runtime = Runtime(portfolio, launch_backend=args.backend)
    producer = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(producer)
    with torch.cuda.stream(stream):
        prepared = runtime.prepare(*inputs)
        for _ in range(args.replays):
            output = prepared.launch()
        expected = operation_reference(portfolio["operation"])(*inputs)
    producer.wait_stream(stream)
    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
    print(
        json.dumps(
            {
                "shape": list(shape),
                "selection": prepared.selection,
                "output_shape": list(output.shape),
                "correctness": "passed",
                "preparation_wall_s": prepared.preparation_wall_s,
            },
            indent=2,
        )
    )
    runtime.close()


if __name__ == "__main__":
    main()
