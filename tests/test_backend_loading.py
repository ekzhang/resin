import pytest
import torch

from resin.backend.loading import load_safetensor_weights, prepare_model, resolve_dtype


def test_resolve_dtype_accepts_torch_dtypes_names_and_none() -> None:
    assert resolve_dtype(None) is None
    assert resolve_dtype(torch.bfloat16) is torch.bfloat16
    assert resolve_dtype("bfloat16") is torch.bfloat16
    assert resolve_dtype("torch.bfloat16") is torch.bfloat16


def test_resolve_dtype_rejects_unknown_names_and_types() -> None:
    with pytest.raises(ValueError, match="unknown torch dtype"):
        resolve_dtype("not_a_dtype")
    with pytest.raises(TypeError, match="dtype must be"):
        resolve_dtype(16)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_load_safetensor_weights_streams_and_casts(tmp_path) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    expected_weight = torch.arange(6, dtype=torch.float32).view(2, 3)
    expected_bias = torch.tensor([7.0, 8.0])
    safetensors.save_file(
        {
            "weight": expected_weight,
            "bias": expected_bias,
        },
        tmp_path / "model.safetensors",
    )
    model = torch.nn.Linear(3, 2, device="meta")

    load_safetensor_weights(
        tmp_path,
        model,
        device="cpu",
        dtype="float64",
    )

    assert model.weight.dtype is torch.float64
    torch.testing.assert_close(model.weight, expected_weight.double())
    torch.testing.assert_close(model.bias, expected_bias.double())


def test_load_safetensor_weights_rejects_missing_model_state(tmp_path) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    safetensors.save_file({"weight": torch.ones(2, 3)}, tmp_path / "model.safetensors")

    with pytest.raises(ValueError, match="missing tensors"):
        load_safetensor_weights(
            tmp_path,
            torch.nn.Linear(3, 2, device="meta"),
            device="cpu",
        )


def test_preparation_preserves_aliases_and_can_fuse_loaded_children() -> None:
    shared = torch.nn.Linear(3, 3, bias=False)
    model = torch.nn.ModuleList([shared, shared, torch.nn.Sequential(shared)])
    hidden = torch.randn(2, 3)
    expected = shared(hidden)
    lowered = []

    def lower(module):
        if isinstance(module, torch.nn.Linear):
            lowered.append(module)
            replacement = torch.nn.Linear(3, 3, bias=False)
            replacement.weight = module.weight
            return replacement
        if isinstance(module, torch.nn.Sequential):
            return module[0]
        return module

    runtime = prepare_model(model, lower)
    assert isinstance(runtime, torch.nn.ModuleList)

    assert lowered == [shared]
    assert runtime[0] is runtime[1] is runtime[2]
    assert runtime[0].weight is shared.weight
    torch.testing.assert_close(runtime[2](hidden), expected)


def test_preparation_releases_consumed_weight_storage_between_layers() -> None:
    import weakref

    class Checkpoint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(3, 3, bias=False)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn = Checkpoint()

    first, second = Layer(), Layer()
    model = torch.nn.ModuleList([first, second])
    old_weight = weakref.ref(first.ffn.projection.weight)
    converted = 0

    def lower(module):
        nonlocal converted
        if isinstance(module, Checkpoint):
            if converted:
                assert old_weight() is None
            converted += 1
            return torch.nn.Identity()
        return module

    prepare_model(model, lower)
    assert converted == 2
    assert old_weight() is None
