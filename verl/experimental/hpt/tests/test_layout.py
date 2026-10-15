import pytest

from verl.experimental.hpt.layout import get_hpt_padding_divisor


@pytest.mark.parametrize(
    "dp_size, expected",
    [
        (1, 8),
        (8, 8),
        (16, 16),
        (48, 48),
        (6, 24),
    ],
)
def test_hpt_padding_divisor_supports_multi_node_data_parallelism(dp_size, expected):
    assert get_hpt_padding_divisor(dp_size) == expected


def test_hpt_padding_divisor_rejects_nonpositive_dp_size():
    with pytest.raises(ValueError, match="data-parallel size must be positive"):
        get_hpt_padding_divisor(0)
