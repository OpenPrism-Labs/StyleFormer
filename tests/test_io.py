"""Checkpoint safety and namespace regression tests."""

import pickle
from collections import OrderedDict

import pytest
import torch
from torch import nn

from src.utils.io import load_checkpoint, load_pretrained_weights, save_checkpoint


class LegacyMetadata:
    pass


def test_checkpoint_requires_explicit_trust_for_objects(tmp_path):
    path = save_checkpoint({"metadata": LegacyMetadata()}, tmp_path)
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(path)
    assert isinstance(load_checkpoint(path, weights_only=False)["metadata"], LegacyMetadata)


def test_pretrained_prefix_preserves_nested_module_names(tmp_path):
    model = nn.Sequential(OrderedDict([("submodule", nn.Linear(2, 1))]))
    expected = {
        "module." + key: torch.ones_like(value) for key, value in model.state_dict().items()
    }
    path = save_checkpoint({"state_dict": expected}, tmp_path)
    load_pretrained_weights(model, path)
    for value in model.state_dict().values():
        torch.testing.assert_close(value, torch.ones_like(value))


def test_plain_state_roundtrip(tmp_path):
    state = {"weight": torch.arange(4), "epoch": 3}
    loaded = load_checkpoint(save_checkpoint(state, tmp_path))
    torch.testing.assert_close(loaded["weight"], state["weight"])
    assert loaded["epoch"] == 3
