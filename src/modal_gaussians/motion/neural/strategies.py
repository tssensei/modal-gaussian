"""Lazy strategy boundary: current component fields do not import old producers."""
from importlib import import_module

import numpy as np


_LEGACY = {
    "guarded": ("guarded_attachments", "GuardedAttachmentConfig", "build_guarded_controls"),
    "pointwise": ("pointwise_attachments", "PointwiseAttachmentConfig", "build_pointwise_controls"),
    "surface": ("surface_attachments", "SurfaceAttachmentConfig", "build_surface_controls"),
    None: ("training_fragments", "FragmentPropagationConfig", "build_training_controls"),
}


def strategy_module(config):
    if not isinstance(config, dict):
        raise ValueError("Motion strategy configuration must be an object")
    name = config.get("strategy")
    if name == "component_field":
        from . import component_field
        return component_field
    if name not in _LEGACY:
        raise ValueError(f"Unknown motion strategy: {name}")
    return import_module("modal_gaussians.motion.legacy.neural." + _LEGACY[name][0])


def config_class(config):
    module = strategy_module(config)
    name = config.get("strategy")
    return getattr(module, "ComponentFieldConfig" if name == "component_field" else _LEGACY[name][1])


def artifact_contract(config):
    if config is None:
        return 8, "neural_complex_displacement_field"
    module = strategy_module(config)
    return module.VERSION, module.METHOD


def array_names(config):
    return strategy_module(config).ARRAY_NAMES


def build_training_controls(graph, *, geometry_config, fragment_config, scene_scale, attachment_inputs=None):
    module = strategy_module(fragment_config)
    name = fragment_config.get("strategy")
    builder = "build_component_controls" if name == "component_field" else _LEGACY[name][2]
    return getattr(module, builder)(graph, geometry_config=geometry_config,
        fragment_config=fragment_config, scene_scale=scene_scale, attachment_inputs=attachment_inputs)


def _array_strategy(arrays):
    for marker, name in (("u_own_field_mask", "component_field"), ("h_source_mask", "guarded"),
                         ("p_source_mask", "pointwise"), ("a_point_status", "surface")):
        if marker in arrays:
            return strategy_module({"strategy": name})
    return strategy_module({})


def training_support_roles(arrays, observed):
    module = _array_strategy(arrays)
    if "u_own_field_mask" in arrays or "h_source_mask" in arrays or "p_source_mask" in arrays:
        return module.support_roles(arrays, observed)
    return strategy_module({}).training_support_roles(arrays, observed)


def diagnostics(arrays):
    return _array_strategy(arrays).diagnostics(arrays)


def validate_training_controls(arrays, graph, *, geometry_config, fragment_config, scene_scale):
    from .geometry_graph import ControlGraph
    name = fragment_config.get("strategy")
    module = strategy_module(fragment_config)
    if name in ("component_field", "guarded"):
        keys = module.INPUT_NAMES | {"observation_view_mask", "contribution_mass"}
        inputs = {key: arrays[key] for key in keys}
    elif name == "surface":
        inputs = {key: arrays[key] for key in module.INPUT_NAMES}
    elif name == "pointwise":
        inputs = {"p_observation_view_mask": arrays["observation_view_mask"]}
    else:
        inputs = None
    expected = build_training_controls(graph, geometry_config=geometry_config,
        fragment_config=fragment_config, scene_scale=scene_scale, attachment_inputs=inputs)
    for key, reference in expected.items():
        value = arrays[key]
        if value.dtype != reference.dtype or not np.array_equal(value, reference):
            raise ValueError(f"Training geometry/attachment differs: {key}")
    ControlGraph.from_dict({key[2:]: value for key, value in expected.items() if key.startswith("c_")})


def implementation_modules(config):
    """Only implementations contributing to the selected strategy enter its revision."""
    from ..common import graph_ops, point_transfer, visibility
    module = strategy_module(config) if config is not None else None
    if config is None:
        return ()
    if config.get("strategy") == "component_field":
        return (module, graph_ops, point_transfer, visibility)
    # Older producers share helpers with one another. Keep their dependency
    # scope explicit without importing them in a current-baseline run.
    modules = [module, graph_ops, point_transfer, visibility]
    for name in ("training_fragments", "fragment_propagation", "surface_attachments", "pointwise_attachments", "guarded_attachments"):
        dependency = import_module("modal_gaussians.motion.legacy.neural." + name)
        if dependency not in modules:
            modules.append(dependency)
    return tuple(modules)
