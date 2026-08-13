import pytest
from asterion.driver import entry_contract
from asterion.io import ValidationError


def test_bounded_generated_abi_is_explicit():
    ptx = ".visible .entry kernel( .param .u64 p0, .param .u64 p1, .param .u32 n, .param .u64 scratch ) .reqntid 128, 1, 1"
    assert entry_contract(ptx, 2, 1) == ("kernel", 128, 4)


@pytest.mark.parametrize("ptx", ["not ptx", ".visible .entry f(.param .u64 x) .reqntid 128, 1, 1"])
def test_unknown_abi_rejected(ptx):
    with pytest.raises(ValidationError):
        entry_contract(ptx, 2, 1)
