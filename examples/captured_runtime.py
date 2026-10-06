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


