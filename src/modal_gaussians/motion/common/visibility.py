"""Frozen center visibility using the calibrated camera and saved depth tolerances."""
import numpy as np

def observation_inputs(points, cameras, depths, alphas, tolerances, alpha_minimum):
    """Freeze center visibility and the same Jacobian used by full-image training."""
    from modal_gaussians.camera_geometry import project_camera
    from modal_gaussians.motion.common.projection import projection_jacobian
    points = np.asarray(points, np.float64)
    visible = np.zeros((len(points), len(cameras)), bool)
    jacobians = []
    for v, camera in enumerate(cameras):
        K, transform = camera.K.cpu().numpy(), camera.world_to_camera.cpu().numpy()
        xyz = points @ transform[:3, :3].T + transform[:3, 3]
        J, projectable = projection_jacobian(points, K, transform, camera.radial_distortion)
        jacobians.append(J)
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = project_camera(xyz, K, camera.radial_distortion)
        valid = projectable & np.isfinite(uv).all(1) & (xyz[:, 2] > 0)
        xy = np.zeros((len(points), 2), np.int64)
        xy[valid] = np.rint(uv[valid]).astype(np.int64)
        valid &= (xy >= 0).all(1) & (xy[:, 0] < depths[v].shape[1]) & (xy[:, 1] < depths[v].shape[0])
        rows = np.flatnonzero(valid)
        depth = depths[v][xy[rows, 1], xy[rows, 0]]
        visible[rows, v] = ((tolerances[v] > 0) & np.isfinite(depth) & (depth > 0)
            & (np.abs(xyz[rows, 2]-depth) <= tolerances[v])
            & (alphas[v][xy[rows, 1], xy[rows, 0]] >= alpha_minimum))
    return {"h_surface_visible": visible, "h_view_jacobian": np.stack(jacobians).astype(np.float32)}

