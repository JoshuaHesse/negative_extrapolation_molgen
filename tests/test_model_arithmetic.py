import math

import pytest
import torch
from torch import nn

from neon_molgen.model import negative_extrapolate, parameter_update_norm


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(2, 2, bias=False)
        self.output = nn.Linear(2, 1, bias=False)


def filled_model(first: float, output: float) -> TinyModel:
    model = TinyModel()
    with torch.no_grad():
        model.first.weight.fill_(first)
        model.output.weight.fill_(output)
    return model


def test_negative_extrapolation_reverses_only_selected_scope() -> None:
    base = filled_model(0.0, 0.0)
    bad = filled_model(2.0, 3.0)

    edited = negative_extrapolate(
        base,
        bad,
        0.5,
        include_patterns=["output"],
    )

    assert torch.equal(edited.first.weight, base.first.weight)
    assert torch.allclose(edited.output.weight, torch.full_like(edited.output.weight, -1.5))


def test_norm_matched_direction_has_reference_update_norm() -> None:
    base = filled_model(0.0, 0.0)
    random_tuned = filled_model(1.0, 1.0)
    bad_tuned = filled_model(2.0, 2.0)

    edited = negative_extrapolate(
        base,
        random_tuned,
        1.0,
        norm_match_to=bad_tuned,
    )

    applied_norm = parameter_update_norm(base, edited)
    reference_norm = parameter_update_norm(base, bad_tuned)
    assert math.isclose(applied_norm, reference_norm, rel_tol=1e-6)


def test_zero_direction_cannot_be_norm_matched() -> None:
    base = filled_model(0.0, 0.0)
    bad_tuned = filled_model(1.0, 1.0)

    with pytest.raises(ValueError, match="zero parameter-update direction"):
        negative_extrapolate(base, base, 1.0, norm_match_to=bad_tuned)
