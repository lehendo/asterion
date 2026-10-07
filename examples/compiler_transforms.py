"""CPU-only inspection of an Asterion-owned schedule rewrite and generated source."""

import argparse
from pathlib import Path

from asterion.codegen import generate
from asterion.io import atomic_json
from asterion.ir import Schedule, make_program, transform

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
program = make_program("softmax", (17, 127))
before = Schedule(row_tile=128)
after, audit = transform(program, before, rows=4, warps=8)
atomic_json(out / "schedule-rewrite.json", audit)
(out / "before.py").write_text(generate(program, before))
(out / "after.py").write_text(generate(program, after))
print(out / "schedule-rewrite.json")
