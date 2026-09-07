"""Defaults for new experiments, separate from historical artifact decoding."""
from .component_field import ComponentFieldConfig


NEURAL_OVERRIDES = {
    "hidden_dim": 256,
    "local_feature_dim": 32,
    "message_layers": 3,
    "control_radius_fraction": 0.015,
    "max_controls": 32768,
    "graph_neighbors": 8,
    "graph_max_distance": 0.008,
    "graph_edge_filter": "none",
    "deformation_weight": 0.1,
    "rotation_weight": 0.1,
}


def baseline_overrides():
    """Override motion representation while retaining prepared observation units."""
    return {"neural": dict(NEURAL_OVERRIDES), "fragment": ComponentFieldConfig().to_dict()}


def new_training_config():
    from .neural_modes import NeuralModesConfig
    return NeuralModesConfig(**NEURAL_OVERRIDES,
        training_fragment_config=ComponentFieldConfig().to_dict())
