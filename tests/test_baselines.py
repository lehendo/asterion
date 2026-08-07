import ast

import pytest
from asterion.baselines import tunable_row_source
from asterion.io import ValidationError


@pytest.mark.parametrize("operation", ["sum", "max", "softmax"])
def test_official_adapter_is_parameterized_generated_ir(operation):
    source = tunable_row_source(operation)
    ast.parse(source)
    assert "ROW_TILE: tl.constexpr" in source
    assert "ROW_GROUP: tl.constexpr" in source
    assert "tl.where(mask" in source


def test_unsupported_official_adapter_is_explicit():
    with pytest.raises(ValidationError):
        tunable_row_source("gemm")
