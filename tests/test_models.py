import torch
import pytest

from wind_repro.models import (
    DLinear,
    EmaFrequencyGraphNet,
    ITransformer,
    MSTNet,
    CommonResidualGraphNet,
    DLinearGeo,
    DLinearSharedResidualMLPGeo,
    MODEL_REGISTRY,
    PatchTST,
    PatchCommonResidualNet,
    TCN,
    TimeMixer,
    TemporalTransformer,
    build_model,
)


def test_dlinear_shape_and_backward() -> None:
    model = DLinear(
        sequence_length=24,
        prediction_length=6,
        num_features=8,
        moving_average=7,
        individual=True,
    )
    features = torch.randn(5, 24, 8)
    target = torch.randn(5, 6, 8)
    prediction = model(features)
    assert prediction.shape == target.shape
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_three_layer_comparison_models_shape_and_backward() -> None:
    config = {
        "data": {"lookback": 24, "horizon": 6},
        "model": {
            "patch_length": 6,
            "stride": 3,
            "d_model": 16,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 32,
            "hidden_size": 32,
            "scale_factors": [1, 2, 4],
            "dropout": 0.0,
            "revin": True,
            "use_norm": True,
        },
    }
    for model_name, expected_type in (
        ("patchtst", PatchTST),
        ("itransformer", ITransformer),
        ("timemixer", TimeMixer),
    ):
        model = build_model(model_name, config, num_features=6)
        assert isinstance(model, expected_type)
        features = torch.randn(2, 24, 6)
        prediction = model(features)
        assert prediction.shape == (2, 6, 6)
        prediction.square().mean().backward()
        assert all(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        assert getattr(model, "blocks", getattr(model, "encoder", None))


def test_dlinear_rejects_even_moving_average() -> None:
    try:
        DLinear(
            sequence_length=12,
            prediction_length=3,
            num_features=2,
            moving_average=4,
        )
    except ValueError as error:
        assert "positive odd integer" in str(error)
    else:
        raise AssertionError("Expected an invalid moving-average error.")


def test_dlinear_is_available_through_model_registry() -> None:
    config = {
        "data": {"lookback": 12, "horizon": 3},
        "model": {"moving_average": 5, "individual": False},
    }
    model = build_model("DLinear", config, num_features=4)
    assert isinstance(model, DLinear)
    assert "dlinear" in MODEL_REGISTRY


@pytest.mark.parametrize(
    ("model_name", "decomposition", "use_graph"),
    [
        ("dlinear_geo", "moving_average", True),
        ("dlinear_ema_geo", "ema", True),
        ("dlinear_ema", "ema", False),
    ],
)
def test_dlinear_geo_variants_shape_backward_and_direction(
    model_name: str, decomposition: str, use_graph: bool
) -> None:
    config = {
        "data": {"lookback": 24, "horizon": 6},
        "model": {
            "decomposition": decomposition,
            "moving_average": 7,
            "ema_alpha": 0.2,
            "graph_k": 2,
            "graph_strength": 0.5,
            "use_graph": use_graph,
        },
    }
    prepared = {
        "metadata": {
            "num_turbines": 3,
            "direction_representation": "scalar",
            "turbines": ["WT01", "WT02", "WT04"],
        },
        "coordinates": (
            torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
            .numpy()
        ),
        "speed_mean": torch.zeros(3).numpy(),
        "speed_std": torch.ones(3).numpy(),
        "direction_mean": torch.zeros(3).numpy(),
        "direction_std": torch.ones(3).numpy(),
    }
    model = build_model(
        model_name,
        config,
        num_features=6,
        prepared=prepared,
    )
    assert isinstance(model, DLinearGeo)
    features = torch.randn(2, 24, 6)
    target = torch.randn(2, 6, 6)
    prediction = model(features)
    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()
    torch.nn.functional.mse_loss(prediction, target).backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert {"dlinear_geo", "dlinear_ema_geo", "dlinear_ema"}.issubset(
        MODEL_REGISTRY
    )


def test_dlinear_geo_requires_coordinates_when_graph_is_enabled() -> None:
    config = {
        "data": {"lookback": 12, "horizon": 3},
        "model": {"use_graph": True},
    }
    prepared = {
        "metadata": {
            "num_turbines": 2,
            "direction_representation": "scalar",
        },
        "direction_mean": torch.zeros(2).numpy(),
        "direction_std": torch.ones(2).numpy(),
    }
    with pytest.raises(ValueError, match="requires finite coordinates"):
        build_model("dlinear_geo", config, num_features=4, prepared=prepared)


def test_dlinear_mlp_geo_shape_and_backward() -> None:
    config = {
        "data": {"lookback": 24, "horizon": 6},
        "model": {
            "moving_average": 7,
            "graph_k": 2,
            "residual_hidden_size": 8,
            "residual_dropout": 0.0,
            "use_graph": True,
        },
    }
    prepared = {
        "metadata": {
            "num_turbines": 3,
            "direction_representation": "scalar",
        },
        "coordinates": (
            torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
            .numpy()
        ),
        "speed_mean": torch.zeros(3).numpy(),
        "speed_std": torch.ones(3).numpy(),
        "direction_mean": torch.zeros(3).numpy(),
        "direction_std": torch.ones(3).numpy(),
    }
    model = build_model(
        "dlinear_mlp_geo",
        config,
        num_features=6,
        prepared=prepared,
    )
    assert isinstance(model, DLinearSharedResidualMLPGeo)
    features = torch.randn(2, 24, 6)
    target = torch.randn(2, 6, 6)
    prediction = model(features)
    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()
    torch.nn.functional.mse_loss(prediction, target).backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert "dlinear_mlp_geo" in MODEL_REGISTRY


def test_mstnet_shape_and_backward() -> None:
    config = {
        "data": {"lookback": 24, "horizon": 6},
        "model": {
            "num_scales": 2,
            "top_components": 2,
            "hidden_size": 8,
            "upstream_turbines": 2,
            "leading_segments": 2,
            "dropout": 0.0,
        },
    }
    prepared = {
        "metadata": {"num_turbines": 3},
        "coordinates": torch.tensor(
            [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]
        ).numpy(),
        "direction_mean": torch.zeros(3).numpy(),
        "direction_std": torch.ones(3).numpy(),
    }
    model = build_model("mstnet", config, num_features=6, prepared=prepared)
    assert isinstance(model, MSTNet)
    features = torch.randn(2, 24, 6)
    target = torch.randn(2, 6, 6)
    prediction = model(features)
    assert prediction.shape == target.shape
    torch.nn.functional.mse_loss(prediction, target).backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert "mstnet" in MODEL_REGISTRY


def test_mstnet_screened_periods_shape_and_backward() -> None:
    config = {
        "data": {"lookback": 24, "horizon": 6},
        "model": {
            "num_scales": 2,
            "top_components": 2,
            "candidate_periods": [6, 12],
            "period_top_k": 2,
            "hidden_size": 8,
            "upstream_turbines": 2,
            "leading_segments": 2,
            "dropout": 0.0,
        },
    }
    prepared = {
        "metadata": {"num_turbines": 3},
        "coordinates": (
            torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
            .numpy()
        ),
        "direction_mean": torch.zeros(3).numpy(),
        "direction_std": torch.ones(3).numpy(),
    }
    model = build_model("mstnet", config, num_features=6, prepared=prepared)
    outputs = model(torch.randn(2, 24, 6))
    assert outputs.shape == (2, 6, 6)
    outputs.mean().backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_phase_common_residual_transformer_shape_and_backward() -> None:
    config = {
        "data": {"lookback": 24, "horizon": 6},
        "model": {
            "screened_periods": [6, 12],
            "residual_encoder": "transformer",
            "residual_d_model": 8,
            "residual_nhead": 2,
            "residual_layers": 1,
            "residual_ffn_size": 16,
            "graph_k": 2,
            "dropout": 0.0,
        },
    }
    prepared = {
        "metadata": {
            "num_turbines": 3,
            "direction_representation": "scalar",
        },
        "coordinates": (
            torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
            .numpy()
        ),
        "speed_mean": torch.zeros(3).numpy(),
        "speed_std": torch.ones(3).numpy(),
        "direction_mean": torch.zeros(3).numpy(),
        "direction_std": torch.ones(3).numpy(),
    }
    model = build_model(
        "phase_common_residual_transformer",
        config,
        num_features=6,
        prepared=prepared,
    )
    outputs = model(torch.randn(2, 24, 6))
    assert outputs.shape == (2, 6, 6)
    outputs.mean().backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_common_residual_graph_shape_and_backward() -> None:
    config = {
        "data": {"lookback": 24, "horizon": 24},
        "model": {
            "ema_alpha": 0.15,
            "graph_k": 2,
            "direction_convention": "from",
        },
    }
    prepared = {
        "metadata": {
            "num_turbines": 3,
            "direction_representation": "scalar",
        },
        "coordinates": (
            torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
            .numpy()
        ),
        "speed_mean": torch.zeros(3).numpy(),
        "speed_std": torch.ones(3).numpy(),
        "direction_mean": torch.zeros(3).numpy(),
        "direction_std": torch.ones(3).numpy(),
    }
    model = build_model(
        "common_residual_graph", config, num_features=6, prepared=prepared
    )
    assert isinstance(model, CommonResidualGraphNet)
    features = torch.randn(2, 24, 6)
    target = torch.randn(2, 24, 6)
    prediction = model(features)
    assert prediction.shape == target.shape
    torch.nn.functional.mse_loss(prediction, target).backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert {"common_residual_graph", "crd_linear"}.issubset(MODEL_REGISTRY)


@pytest.mark.parametrize(
    ("model_name", "model_config"),
    [
        ("patch_linear", {"use_graph": False}),
        ("patch_linear_gated", {"use_graph": False}),
        ("patch_gated", {"use_graph": True}),
        ("patch_standard", {"use_graph": True}),
        ("patch_talking_head", {"use_graph": True}),
        ("patch_ssm", {"use_graph": True}),
    ],
)
def test_patch_common_variants_shape_and_backward(
    model_name: str, model_config: dict
) -> None:
    config = {
        "data": {"lookback": 24, "horizon": 6},
        "model": {
            "patch_length": 4,
            "stride": 2,
            "d_model": 16,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
            **model_config,
        },
    }
    prepared = {
        "metadata": {
            "num_turbines": 3,
            "direction_representation": "scalar",
        },
        "coordinates": (
            torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
            .numpy()
        ),
        "speed_mean": torch.zeros(3).numpy(),
        "speed_std": torch.ones(3).numpy(),
        "direction_mean": torch.zeros(3).numpy(),
        "direction_std": torch.ones(3).numpy(),
    }
    model = build_model(
        model_name,
        config,
        num_features=6,
        prepared=prepared,
    )
    assert isinstance(model, PatchCommonResidualNet)
    features = torch.randn(2, 24, 6)
    target = torch.randn(2, 6, 6)
    prediction = model(features)
    assert prediction.shape == target.shape
    torch.nn.functional.mse_loss(prediction, target).backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@pytest.mark.parametrize(
    ("model_name", "model_config", "expected_type"),
    [
        (
            "ema_freq_graph",
            {
                "ema_alpha": 0.2,
                "graph_k": 2,
                "graph_strength": 0.25,
            },
            EmaFrequencyGraphNet,
        ),
        (
            "transformer",
            {
                "d_model": 16,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 32,
                "dropout": 0.0,
            },
            TemporalTransformer,
        ),
        (
            "tcn",
            {
                "channels": 8,
                "kernel_size": 3,
                "levels": 2,
                "dropout": 0.0,
            },
            TCN,
        ),
        (
            "patchtst",
            {
                "patch_length": 4,
                "stride": 2,
                "d_model": 16,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 32,
                "dropout": 0.0,
                "revin": True,
            },
            PatchTST,
        ),
        (
            "itransformer",
            {
                "d_model": 16,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 32,
                "dropout": 0.0,
                "use_norm": True,
            },
            ITransformer,
        ),
    ],
)
def test_registered_forecasters_shape_and_backward(
    model_name: str,
    model_config: dict,
    expected_type: type[torch.nn.Module],
) -> None:
    config = {
        "data": {"lookback": 12, "horizon": 3},
        "model": model_config,
    }
    model = build_model(model_name, config, num_features=4)
    features = torch.randn(2, 12, 4)
    target = torch.randn(2, 3, 4)
    prediction = model(features)
    assert isinstance(model, expected_type)
    assert prediction.shape == target.shape
    torch.nn.functional.mse_loss(prediction, target).backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
