"""Explicit static-buffer CUDA Graph integration for an existing certified runtime.

Graph replay follows PyTorch's static-address contract and bypasses Python launch
checks. Keep all tensors, prepared invocations and modules alive and unchanged.
Copy new finite data into existing inputs; never rebind or resize their storage.
"""

import argparse
import json

from asterion.backend import make_inputs, require_gpu
from asterion.gpu import operation_reference
from asterion.io import read
from asterion.ir import make_program
from asterion.runtime import Runtime


def capture_prepared(prepared):
    torch, _ = require_gpu()
    producer = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(producer)
    with torch.cuda.stream(stream):
        for _ in range(3):
            prepared.launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = prepared.launch()
    producer.wait_stream(stream)
    return graph, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("portfolio")
    parser.add_argument("--backend", choices=["triton", "native"], default="triton")
    args = parser.parse_args()
    portfolio = read(args.portfolio)
    if not portfolio.get("validation_certificates"):
        parser.error("example requires at least one numerical certificate")
    shape = portfolio["validation_certificates"][0]["shape"]
    torch, _ = require_gpu()
    inputs = make_inputs(make_program(portfolio["operation"], shape), 1729)
    runtime = Runtime(portfolio, launch_backend=args.backend)
    prepared = runtime.prepare(*inputs)
    graph, output = capture_prepared(prepared)
    graph.replay()
    expected = operation_reference(portfolio["operation"])(*inputs)
    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
    print(
        json.dumps(
            {
                "correctness": "passed",
                "selection": prepared.selection,
                "boundary": "standard PyTorch graph replay; static buffers, no per-replay metadata check",
            }
        )
    )
    del graph
    runtime.close()


if __name__ == "__main__":
    main()
