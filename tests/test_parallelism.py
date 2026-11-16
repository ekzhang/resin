from __future__ import annotations

import pickle
from importlib import import_module

import pytest

from resin import Parallelism
from resin.backend.topology import ParallelismCapabilities


def test_topology_composes_all_parallel_axes_without_multiplying_ep():
    topology = Parallelism(world_size=8, cp=2, attn_dp=2, attn_tp=2, ep=4)
    assert topology.world_size == 8
    assert topology.attn_tp == topology.expert_tp == 2
    groups = topology.partitions()
    assert groups["attn_tp"] == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert groups["cp"] == ((0, 2), (1, 3), (4, 6), (5, 7))
    assert groups["attn_dp"] == ((0, 4), (1, 5), (2, 6), (3, 7))
    assert groups["ep"] == ((0, 2, 4, 6), (1, 3, 5, 7))
    assert groups["expert_tp"] == groups["attn_tp"]
    overlap = Parallelism(world_size=8, attn_tp=8, ep=8).partitions()
    assert overlap["attn_tp"] == overlap["ep"] == (tuple(range(8)),)
    assert overlap["expert_tp"] == tuple((rank,) for rank in range(8))
    assert groups["drivers"] == ((0, 4),)
    assert groups["domain"] == (tuple(range(8)),)
    for rank in range(8):
        assert topology.rank(*topology.coordinates(rank)) == rank
    shared = Parallelism(world_size=8, cp=2, attn_dp=2, attn_tp=2).partitions()
    assert shared["expert_tp"] == shared["domain"] == (tuple(range(8)),)
    assert shared["drivers"] == ((0, 4),)


@pytest.mark.parametrize(
    "options,replication,expert",
    [
        ({}, 1, 1),
        ({"world_size": 8}, 8, 8),
        ({"world_size": 8, "attn_tp": 8}, 1, 8),
        ({"world_size": 8, "ep": 8}, 8, 1),
        ({"world_size": 4, "attn_dp": 4}, 1, 4),
        ({"world_size": 2, "attn_tp": 2}, 1, 2),
        ({"world_size": 8, "attn_dp": 8, "ep": 8}, 1, 1),
        ({"world_size": 12, "attn_dp": 3, "cp": 2, "ep": 4}, 2, 3),
        ({"world_size": 16, "attn_dp": 2, "cp": 2, "attn_tp": 2}, 2, 16),
    ],
)
def test_attention_replication_and_expert_tp_cover_the_same_ranks(
    options, replication, expert
):
    topology = Parallelism(**options)
    assert topology.attn_replication == replication and topology.expert_tp == expert
    assert (
        topology.attn_dp * topology.cp * topology.attn_replication * topology.attn_tp
        == topology.world_size
    )
    assert topology.ep * topology.expert_tp == topology.world_size
    for rank in range(topology.world_size):
        assert topology.rank(*topology.coordinates(rank)) == rank
    groups = topology.partitions()
    for name, partitions in groups.items():
        if name != "drivers":
            assert sorted(rank for group in partitions for rank in group) == list(
                range(topology.world_size)
            )
    assert len(groups["replica"]) == topology.attn_dp
    assert len(groups["drivers"][0]) == topology.attn_dp


def test_remaining_attention_ranks_replicate_the_same_request():
    topology = Parallelism(world_size=16, attn_dp=2, cp=2, attn_tp=2, ep=4)
    groups = topology.partitions()
    assert groups["attn_tp"][:2] == ((0, 1), (2, 3))
    assert groups["attn_replication"][:2] == ((0, 2), (1, 3))
    assert groups["cp"][:2] == ((0, 4), (1, 5))
    assert groups["attn_dp"][:2] == ((0, 8), (1, 9))
    assert groups["replica"] == (tuple(range(8)), tuple(range(8, 16)))
    assert groups["drivers"] == ((0, 8),)
    assert groups["expert_tp"] == tuple(
        tuple(range(start, start + 4)) for start in range(0, 16, 4)
    )


@pytest.mark.parametrize("configuration", (Parallelism, ParallelismCapabilities))
def test_parallel_configuration_requires_keywords(configuration):
    with pytest.raises(TypeError):
        configuration(2)


@pytest.mark.parametrize(
    "options",
    [
        {"world_size": 8, "ep": 3},
        {"world_size": 8, "cp": 3},
        {"world_size": 8, "attn_dp": 3},
        {"world_size": 8, "attn_tp": 3},
        {"world_size": 12, "attn_dp": 4, "cp": 2},
        {"world_size": 4, "attn_dp": 2, "cp": 2, "attn_tp": 2},
        {"cp": 0},
        {"world_size": True},
        {"attn_dp": 1.5},
        {"attn_tp": 0},
        {"attn_tp": True},
    ],
)
def test_invalid_topologies_are_rejected(options):
    with pytest.raises(ValueError):
        Parallelism(**options)


def test_parallelism_is_a_native_immutable_value():
    value = Parallelism(world_size=8, attn_tp=2, ep=4)
    equal = Parallelism(world_size=8, attn_tp=2, ep=4)
    assert type(value).__module__ == "resin._resin"
    assert value == equal and value != Parallelism()
    assert len({value, equal}) == 1
    assert repr(value) == (
        "Parallelism(world_size=8, attn_dp=1, attn_tp=2, cp=1, ep=4)"
    )
    for name in (
        "world_size",
        "attn_dp",
        "attn_tp",
        "cp",
        "ep",
        "expert_tp",
        "attn_replication",
    ):
        with pytest.raises(AttributeError):
            setattr(value, name, 2)


@pytest.mark.parametrize("protocol", range(2, pickle.HIGHEST_PROTOCOL + 1))
def test_parallelism_survives_process_transport(protocol):
    value = Parallelism(world_size=16, attn_dp=2, attn_tp=2, cp=2, ep=4)
    restored = pickle.loads(pickle.dumps(value, protocol=protocol))
    assert restored == value and hash(restored) == hash(value)
    assert restored.partitions() == value.partitions()


@pytest.mark.parametrize("field", ("world_size", "attn_dp", "attn_tp", "cp", "ep"))
@pytest.mark.parametrize("value", (0, -1, True, 1.5, 1 << 128))
def test_parallelism_sizes_are_strict_positive_integers(field, value):
    with pytest.raises(ValueError):
        Parallelism(**{field: value})


@pytest.mark.parametrize("rank", (-1, 8, True, 1.5, 1 << 128))
def test_parallelism_rejects_invalid_rank(rank):
    with pytest.raises(ValueError):
        Parallelism(world_size=8).coordinates(rank)


@pytest.mark.parametrize(
    "coordinates",
    ((-1, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2), (False, 0, 0, 0)),
)
def test_parallelism_rejects_invalid_coordinates(coordinates):
    with pytest.raises(ValueError):
        Parallelism(world_size=16, attn_dp=2, attn_tp=2, cp=2).rank(*coordinates)


def test_capabilities_declare_axes_and_reject_unsupported_combinations():
    single_gpu = ParallelismCapabilities()
    single_gpu.validate(Parallelism(), model="Single GPU")
    with pytest.raises(ValueError, match="expert_tp=2"):
        single_gpu.validate(Parallelism(world_size=2), model="Single GPU")
    with pytest.raises(ValueError, match="Single GPU does not support attn_tp=2"):
        single_gpu.validate(Parallelism(world_size=2, attn_tp=2), model="Single GPU")
    tp = ParallelismCapabilities(attn_tp=True, expert_tp=True)
    tp.validate(Parallelism(world_size=4, attn_tp=4), model="TP")
    ParallelismCapabilities(expert_tp=True).validate(
        Parallelism(world_size=4), model="Expert TP"
    )
    with pytest.raises(ValueError, match="attn_dp=4"):
        tp.validate(Parallelism(world_size=4, attn_dp=4), model="TP")
    ep_dp = ParallelismCapabilities(ep=True, attn_dp=True)
    ep_dp.validate(Parallelism(world_size=4, ep=4, attn_dp=4), model="EP/DP")
    ep_dp.validate(Parallelism(world_size=4, ep=4), model="EP/DP")
    with pytest.raises(ValueError, match="expert_tp=2"):
        ep_dp.validate(Parallelism(world_size=4, ep=2, attn_dp=4), model="EP/DP")
    with pytest.raises(ValueError, match="cp=2"):
        ep_dp.validate(Parallelism(world_size=4, ep=4, cp=2), model="EP/DP")


@pytest.mark.parametrize(
    "module,runner",
    [
        ("resin_models_torch.lfm2.reference", "Lfm2TorchRunner"),
    ],
)
def test_unsharded_model_implementations_reject_attention_tp(module, runner):
    implementation = getattr(import_module(module), runner)
    with pytest.raises(ValueError, match="does not support attn_tp=2"):
        implementation.parallelism_capabilities.validate(
            Parallelism(world_size=2, attn_tp=2), model=runner
        )


def test_inkling2_supports_combined_attention_tp_dp_and_expert_parallelism():
    from resin_models_cuda.inkling2.cuda import Inkling2CudaRunner

    capabilities = Inkling2CudaRunner.parallelism_capabilities
    capabilities.validate(
        Parallelism(world_size=4, attn_tp=2, attn_dp=2, ep=4), model="Inkling 2"
    )
    for topology in (
        Parallelism(world_size=2, attn_tp=2, ep=1),
        Parallelism(world_size=4, attn_tp=2, ep=2),
    ):
        capabilities.validate(topology, model="Inkling 2")


@pytest.mark.parametrize(
    "module,runner",
    [
        ("resin_models_cuda.gemma4.cuda", "Gemma4CudaRunner"),
        ("resin_models_cuda.qwen3_5.cuda", "Qwen3_5CudaRunner"),
    ],
)
def test_dense_cuda_models_shard_ffns_with_optional_attention_tp(module, runner):
    capabilities = getattr(import_module(module), runner).parallelism_capabilities
    for attention in (1, 2):
        capabilities.validate(
            Parallelism(world_size=2, attn_tp=attention), model=runner
        )
    for axis in ("attn_dp", "ep", "cp"):
        with pytest.raises(ValueError, match=f"does not support {axis}=2"):
            capabilities.validate(Parallelism(world_size=2, **{axis: 2}), model=runner)
