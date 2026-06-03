from typing import Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


def adjust_min_len(xs: tuple[Any]) -> tuple[Any]:
    """
    fix frame length to minimum length in xs

    Args:
        xs: features need shape (T, ) or (T, D)

    Returns:
        fixed length features
    """
    min_len_x = min([x.shape[0] for x in xs])
    max_len_x = max([x.shape[0] for x in xs])
    # assert max_len_x - min_len_x <= 1, (
    #     f"The difference between frames for each feature is too large. {max_len_x - min_len_x} frames"
    # )
    xs = [x[:min_len_x] if x.ndim == 1 else x[:min_len_x, :] for x in xs]

    return xs


def check_length(xs: tuple[Any]) -> bool:
    """
    check if all features have the same length

    Args:
        xs: features need shape (T, ) or (T, D)
    """

    xs_len = [x.shape[0] for x in xs]

    return all(xs_len)
