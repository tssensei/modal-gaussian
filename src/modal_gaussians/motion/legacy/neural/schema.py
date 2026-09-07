"""Persisted per-mode array names; imports no training implementation."""

POINTWISE_MODE_ARRAYS = {"p_observation_view_mask", "p_source_mask", "p_neighbor_index",
               "p_neighbor_weight", "p_motion_component_index"}

GUARDED_MODE_ARRAYS = {"h_source_mask", "h_residual_mask", "h_reliable_view_mask", "h_residual_projector",
               "h_neighbor_index", "h_neighbor_weight", "h_motion_component_index", "h_control_supported"}

