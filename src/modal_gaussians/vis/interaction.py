"""Viser drag adapter and rasterizer-consistent foreground picking."""
from dataclasses import dataclass
from collections import deque
from queue import Empty, SimpleQueue
# import json
import threading
import time

import cv2
import numpy as np
import torch

from modal_gaussians.motion.interaction import InteractionConfig, ModalInteraction, mode_factors

DRIVE_INTERACTIVE = "Interactive simulation"


def pick_foreground(info, screen, foreground_count, minimum_alpha=0.1):
    """Pick the dominant contributor at a pixel in a single pinhole render.

    Reuse gsplat's tile ordering/conics, including background occlusion and its
    exclusive transmittance cutoff. No approximate surface-depth/nearest search.
    """
    uv = np.asarray(screen, dtype=np.float64)
    if uv.shape != (2,) or not np.isfinite(uv).all() or np.any(uv < 0) or np.any(uv >= 1):
        return None
    width, height, tile = int(info['width']), int(info['height']), int(info['tile_size'])
    x, y = int(uv[0] * width), int(uv[1] * height)
    offsets = info['isect_offsets'].reshape(-1)
    index = (y // tile) * int(info['tile_width']) + x // tile
    start = int(offsets[index])
    end = int(offsets[index + 1]) if index + 1 < len(offsets) else len(info['flatten_ids'])
    ids = info['flatten_ids'][start:end].long()
    if not len(ids):
        return None
    def rows(name, columns):
        return info[name].reshape(-1, columns)[ids].detach().cpu().double().numpy()
    delta = rows('means2d', 2) - (x + 0.5, y + 0.5)
    conics = rows('conics', 3)
    sigma = (0.5 * (conics[:, 0] * delta[:, 0]**2 + conics[:, 2] * delta[:, 1]**2)
             + conics[:, 1] * delta[:, 0] * delta[:, 1])
    alpha = np.minimum(0.999, rows('opacities', 1)[:, 0] * np.exp(-np.maximum(sigma, 0)))
    alpha[(sigma < 0) | (alpha < 1 / 255)] = 0
    trans_after = np.cumprod(1 - alpha)
    weights = alpha * np.concatenate(([1.0], trans_after[:-1]))
    weights[trans_after <= 1e-4] = 0
    ids = ids.cpu().numpy()
    foreground = ids < foreground_count
    if weights[foreground].sum() < minimum_alpha:
        return None
    winner = int(np.argmax(weights))
    return int(ids[winner]) if foreground[winner] and weights[winner] > 0 else None


def screen_on_plane(camera, screen, point):
    """Unproject onto the camera-facing plane through a fixed world point."""
    uv = np.asarray(screen, dtype=np.float64)
    if uv.shape != (2,) or not np.isfinite(uv).all():
        raise ValueError("Invalid pointer coordinates")
    K = camera.K.detach().cpu().double().numpy()
    w2c = camera.world_to_camera.detach().cpu().double().numpy()
    depth = float(w2c[2, :3] @ point + w2c[2, 3])
    if depth <= 0:
        raise ValueError("Grab point is behind the camera")
    ray = np.linalg.solve(K, [uv[0] * camera.width, uv[1] * camera.height, 1.0])
    return w2c[:3, :3].T @ (ray * depth - w2c[:3, 3])


def camera_key(client, resolution):
    return tuple((*client.camera.position, *client.camera.wxyz,
                  client.camera.fov, client.camera.aspect, int(resolution)))


@dataclass
class RenderSnapshot:
    client_id: int
    key: tuple
    camera: object
    z: np.ndarray
    means: torch.Tensor
    info: dict


class ViewerInteraction:
    """One modal session; numerical state is owned by the existing render worker.

    UI callbacks only enqueue messages. The tiny input lock protects gesture
    cancellation tokens; no GPU operation is performed while holding it.
    """

    def __init__(self, viewer):
        self.viewer = viewer
        self.config = InteractionConfig()
        self.active = False
        self.engine = None
        self.snapshot = None
        self.grab = None
        self.events = SimpleQueue()
        self.input_lock = threading.Lock()
        self.pressed = None
        self.serial = 0
        self.planes = {}
        self.saved = []
        self.closed = threading.Event()
        self.metrics = {'picks': 0, 'pick_seconds': 0.0, 'renders': 0,
                        'render_seconds': 0.0, 'publish_seconds': 0.0}
        self.frame_times = deque(maxlen=121)
        self.frame_durations = deque(maxlen=120)
        self.last_stats_update = 0.0
        gui = viewer.server.gui
        with gui.add_folder("Interaction", visible=False, order=-1) as self.folder:
            self.render_fps = gui.add_dropdown("Render FPS", ("30", "60"), initial_value="60")
            self.normalize_modes = gui.add_checkbox("Normalize mode RMS / phase", True)
            self.normalize_support = gui.add_checkbox("Divide by local support (S_p)", True)
            self.damping = gui.add_slider("Damping ratio ζ", min=0., max=1., step=.001,
                                          initial_value=self.config.damping)
            self.strength = gui.add_slider("Interaction strength", min=0., max=5., step=.01,
                                           initial_value=self.config.strength)
            self.maximum_drag_distance = gui.add_slider("Maximum drag distance (%)",
                min=1., max=100., step=1., initial_value=100 * self.config.drag_radius_fraction)
            self.pause = gui.add_checkbox("Pause simulation", False)
            self.reset = gui.add_button("Reset simulation")
            self.status = gui.add_markdown("At rest")
            self.timing = gui.add_markdown("")
        for name, handle in (('damping', self.damping), ('pause', self.pause),
                             ('normalize_modes', self.normalize_modes),
                             ('normalize_support', self.normalize_support),
                             ('render_fps', self.render_fps)):
            @handle.on_update
            async def changed(event, name=name):
                self.enqueue(name, event.target.value)
        @self.reset.on_click
        async def reset(_):
            self.enqueue('reset', None)
        @viewer.server.on_client_disconnect
        def disconnect(client):
            with self.input_lock:
                if self.pressed is not None and self.pressed[0] == client.client_id:
                    self.pressed = None
            self.enqueue('disconnect', client.client_id)
        viewer.drive.options = (*viewer.drive.options, DRIVE_INTERACTIVE)
        self._timer_thread = threading.Thread(target=self._tick, daemon=True)
        self._timer_thread.start()

    def enqueue(self, name, value):
        self.events.put((name, value, time.monotonic()))
        self.viewer.request_render()

    def _tick(self):
        fps = int(self.render_fps.value)
        deadline = time.perf_counter() + 1 / fps
        while not self.closed.is_set():
            now = time.perf_counter()
            target = int(self.render_fps.value)
            if target != fps:
                fps = target
                deadline = now + 1 / fps
            # Windows Event.wait(timeout) rounds this into ~46 ms at 30 Hz.
            # Sleep to an absolute deadline; skip missed displays, never replay them.
            time.sleep(max(0.0, deadline - now))
            if self.closed.is_set():
                break
            now = time.perf_counter()
            if now - deadline >= 1 / fps:
                deadline = now
            deadline += 1 / fps
            engine = self.engine
            if (self.active and engine is not None and not engine.paused
                    and engine.drag_start is None and np.any(engine.z)):
                self.viewer.request_render()

    def _set_enabled(self, enabled):
        v = self.viewer
        self.frame_times.clear()
        self.frame_durations.clear()
        self.last_stats_update = 0.0
        self.snapshot = self.grab = None
        with self.input_lock:
            self.pressed = None
        if enabled:
            if self.engine is None:
                self.status.content = "Preparing interaction modes…"
                fields = (mode.detach().cpu().numpy() for mode in v.data.phi)
                self.normalized_factors = mode_factors(fields)
                self.engine = ModalInteraction(v.data.frequencies_hz, self.normalized_factors,
                                               now=time.monotonic(), config=self.config)
                points = v.data.scene.foreground.active()['means'].detach()
                center = torch.quantile(points, .5, dim=0)
                self.radius = max(float(torch.quantile(torch.linalg.vector_norm(points - center, dim=1), .95)), 1e-6)
                if v.data.rotation is not None and not bool(torch.isfinite(v.data.rotation).all()):
                    self.engine = None
                    raise ValueError("Interaction angular fields must be finite")
            self.active = True
            self.engine.factors = (self.normalized_factors.copy() if self.normalize_modes.value
                                   else (self.normalized_factors != 0).astype(np.complex128))
            self.engine.factors[~self.engine.valid] = 0
            self.engine.reset(time.monotonic())
            self.engine.set_damping(float(self.damping.value), time.monotonic())
            self.engine.set_paused(bool(self.pause.value), time.monotonic())
            changed_values = [(v.canonical, False), (v.follow_camera, False),
                              (v.show_cameras, False),
                              (v.rotate_ellipsoids, v.data.rotation is not None),
                              (v.hide_render, False), (v.color_mode, 'rgb')]
            if v.hide_background is not None:
                changed_values.append((v.hide_background, False))
            handles = [v.playback_view, *v.playback, v.motion_scale, v.disable_all_modes_button]
            handles += [m[key] for m in v.mode_controls for key in ('enabled', 'gain', 'phase')]
            handles += [handle for handle, _ in changed_values]
            self.saved = [(handle, bool(handle.disabled), None) for handle in handles]
            self.saved += [(handle, None, handle.value) for handle, _ in changed_values]
            self.saved += [(v.playback[3], 'visible', v.playback[3].visible),
                           (v.playback[4], 'visible', v.playback[4].visible)]
            for handle in handles:
                handle.disabled = True
            v.playback[3].visible, v.playback[4].visible = False, True
            for handle, value in changed_values:
                handle.value = value
            v._remove_support_cloud()
            v._remove_control_cloud()
            v._remove_component_graph()
            state = 'Paused' if self.engine.paused else 'At rest'
            self.status.content = f"{state} · {np.count_nonzero(self.engine.valid)}/{len(self.engine.valid)} active modes"
        else:
            self.active = False
            if self.engine is not None:
                self.engine.reset(time.monotonic())
            for handle, disabled, value in self.saved:
                if disabled == 'visible':
                    handle.visible = value
                elif disabled is None:
                    handle.value = value
                else:
                    handle.disabled = disabled
            self.saved.clear()
        self.folder.visible = enabled
        for plane, _ in self.planes.values():
            plane.visible = False

    def _plane(self, client):
        key = camera_key(client, self.viewer.viewer_resolution.value)
        old = self.planes.get(client.client_id)
        enabled = self.active
        if old is None:
            plane = client.scene.add_mesh_simple('/interaction/input',
                vertices=np.zeros((4, 3), np.float32), faces=np.array([[0, 1, 2], [0, 2, 3]]),
                opacity=0., side='double', cast_shadow=False, receive_shadow=False,
                visible=False)
            @plane.on_drag('left', modifier='cmd/ctrl')
            async def drag(event):
                if not self.active:
                    return
                with self.input_lock:
                    if event.phase == 'start':
                        if self.pressed is not None:
                            return
                        self.serial += 1
                        self.pressed = (event.client_id, self.serial)
                    token = self.pressed
                    if token is None or token[0] != event.client_id:
                        return
                    if event.phase == 'end':
                        self.pressed = None
                self.viewer._last_client = event.client
                self.enqueue('drag', (event.phase, token, event.start_screen_pos, event.end_screen_pos))
            old = (plane, None)
        plane, previous = old
        if previous != key and self.grab is None:
            # A transparent input surface only; all object depth comes from GS picking.
            depth = max(self.radius, .1)
            half_y = depth * np.tan(float(client.camera.fov) / 2) * 1.05
            half_x = half_y * float(client.camera.aspect)
            plane.vertices = np.array([[-half_x, -half_y, depth], [half_x, -half_y, depth],
                                       [half_x, half_y, depth], [-half_x, half_y, depth]], np.float32)
            plane.position = client.camera.position
            plane.wxyz = client.camera.wxyz
            self.planes[client.client_id] = (plane, key)
        plane.visible = enabled

    def prepare(self, client):
        requested = self.viewer.drive.value == DRIVE_INTERACTIVE
        if requested != self.active:
            self._set_enabled(requested)
        if self.active:
            key = camera_key(client, self.viewer.viewer_resolution.value)
            if self.snapshot is not None and (self.snapshot.client_id != client.client_id or self.snapshot.key != key):
                self.cancel_grab()
                self.snapshot = None
            self._plane(client)
        # Adjacent pointer updates are replaceable; start/end and settings retain order.
        pending = []
        while True:
            try:
                event = self.events.get_nowait()
            except Empty:
                break
            if (pending and event[0] == 'drag' and event[1][0] == 'update'
                    and pending[-1][0] == 'drag' and pending[-1][1][:2] == event[1][:2]):
                pending[-1] = event
            else:
                pending.append(event)
        for name, value, received in pending:
            if not self.active:
                continue
            now = max(received, self.engine.time)
            if name == 'drag':
                self._drag(client, *value, now=now)
            elif name in ('normalize_modes', 'normalize_support'):
                self.cancel_grab(now)
                self.engine.reset(now)
                if name == 'normalize_modes':
                    self.engine.factors = (self.normalized_factors.copy() if value
                                           else (self.normalized_factors != 0).astype(np.complex128))
                    self.engine.factors[~self.engine.valid] = 0
                self.snapshot = None
                self.status.content = 'Paused' if self.engine.paused else 'At rest'
            elif name == 'damping':
                self.engine.set_damping(float(value), now)
            elif name == 'pause':
                self.engine.set_paused(bool(value), now)
                self.status.content = ('Paused' if value else
                                       'Simulating' if np.any(self.engine.z) else 'At rest')
            elif name == 'render_fps':
                self.frame_times.clear()
                self.frame_durations.clear()
                self.last_stats_update = 0.0
            elif name in ('reset', 'disconnect'):
                self.cancel_grab(now)
                if name == 'reset':
                    self.engine.reset(now)
                    self.snapshot = None
                    self.status.content = 'Paused' if self.engine.paused else 'At rest'
                if name == 'disconnect':
                    self.engine.set_paused(True, now)
                    self.pause.value = True
                    self.status.content = 'Paused · client disconnected'
        if self.active:
            self._plane(client)

    def cancel_grab(self, now=None):
        if self.grab is not None:
            self.status.content = 'Paused' if self.engine.paused else 'Simulating'
        self.grab = None
        with self.input_lock:
            self.pressed = None
        if self.engine is not None:
            self.engine.release(time.monotonic() if now is None else now)

    def _drag(self, client, phase, token, start_screen, end_screen, *, now):
        if token[0] != client.client_id:
            return
        if phase == 'start':
            with self.input_lock:
                if self.pressed != token:
                    return
            snap = self.snapshot
            if snap is None or snap.key != camera_key(client, self.viewer.viewer_resolution.value):
                self.status.content = 'Wait for a current rendered frame, then drag again'
                return
            began = time.perf_counter()
            index = pick_foreground(snap.info, start_screen, self.viewer.data.scene.foreground.count,
                                    self.config.foreground_alpha_minimum)
            self.metrics['picks'] += 1
            self.metrics['pick_seconds'] += time.perf_counter() - began
            if index is None:
                self.status.content = 'No visible foreground at this point'
                return
            point = snap.means[index].detach().cpu().double().numpy()
            field = self.viewer.data.phi[:, index].detach().cpu().numpy()
            start_point = screen_on_plane(snap.camera, start_screen, point)
            with self.input_lock:
                if self.pressed != token:
                    return
                if not self.engine.begin_drag(field, snap.z, now,
                                             normalize_support=bool(self.normalize_support.value)):
                    self.status.content = 'This point has no motion support'
                    return
                self.grab = (token, index, snap.camera, point, start_point)
        if self.grab is None or self.grab[0] != token:
            return
        _, index, camera, point, start = self.grab
        d = screen_on_plane(camera, end_screen, point) - start
        maximum = float(self.maximum_drag_distance.value) / 100 * self.radius
        limited = self.engine.drag(d, strength=float(self.strength.value),
                                   maximum=maximum)
        self.drag_delta = d * min(1., maximum / max(np.linalg.norm(d), 1e-12))
        self.status.content = f'Dragging foreground #{index}' + (' · drag limit reached' if limited else '')
        if phase == 'end':
            self.engine.release(now)
            self.grab = None
            self.status.content = 'Paused' if self.engine.paused else 'Simulating'
            print(f"Interaction release: foreground={index}, damping={self.engine.damping}, {self.metrics}")

    def coordinates(self):
        return self.engine.coordinates(time.monotonic())

    def render(self, client):
        v = self.viewer
        q = self.coordinates()
        z = self.engine.z.copy()
        means = v.data.deformed_means(q)
        key = camera_key(client, v.viewer_resolution.value)
        camera = v._render_camera(client)
        output, info = v.data.scene.render_deformed_with_info(camera, means,
            foreground_quaternions=v.data.deformed_quaternions(q), include_background=True)
        frame = output['rgb'].clamp(0, 1).mul(255).round().byte().cpu().numpy()
        self.snapshot = (RenderSnapshot(client.client_id, key, camera, z, means, info)
                         if key == camera_key(client, v.viewer_resolution.value) else None)
        if self.grab is not None:
            _, index, _, point, _ = self.grab
            p = means[index].detach().cpu().numpy()
            w2c = camera.world_to_camera.cpu().numpy()
            xy = camera.K.cpu().numpy() @ (w2c[:3, :3] @ p + w2c[:3, 3])
            if xy[2] > 0:
                pixel = tuple(np.rint(xy[:2] / xy[2] - .5).astype(int))
                cv2.circle(frame, pixel, 5, (255, 210, 30), 2)
            endpoints = np.stack((point, point + self.drag_delta))
            projected = (endpoints @ w2c[:3, :3].T + w2c[:3, 3]) @ camera.K.cpu().numpy().T
            if np.all(projected[:, 2] > 0):
                a, b = np.rint(projected[:, :2] / projected[:, 2:] - .5).astype(int)
                cv2.arrowedLine(frame, tuple(a), tuple(b), (255, 210, 30), 2, tipLength=.2)
        return frame

    def note_frame(self, render_seconds, publish_seconds, finished):
        """Measure completed JPEG/message enqueue, not browser presentation."""
        self.metrics['renders'] += 1
        self.metrics['render_seconds'] += render_seconds
        self.metrics['publish_seconds'] += publish_seconds
        dragging = self.engine.drag_start is not None
        idle = self.engine.paused or not np.any(self.engine.z)
        if dragging or idle or (self.frame_times and finished - self.frame_times[-1] > 1):
            self.frame_times.clear()
            self.frame_durations.clear()
        if dragging or idle:
            self.metrics['server_fps'] = 0.0
            self.timing.content = ('Dragging · updates on input' if dragging else
                                   'Paused' if self.engine.paused else 'Idle')
            self.last_stats_update = 0.0
            return
        self.frame_times.append(finished)
        self.frame_durations.append((render_seconds, publish_seconds))
        elapsed = finished - self.frame_times[0]
        if finished - self.last_stats_update < 1 or elapsed < 1:
            return
        self.last_stats_update = finished
        times = np.asarray(self.frame_durations) * 1000
        total = times.sum(axis=1)
        self.metrics.update(server_fps=(len(self.frame_times) - 1) / elapsed,
            target_fps=int(self.render_fps.value),
            render_ms=float(np.median(times[:, 0])), publish_ms=float(np.median(times[:, 1])),
            total_ms=float(np.median(total)), total_p95_ms=float(np.percentile(total, 95)),
            interval_p95_ms=float(np.percentile(np.diff(self.frame_times) * 1000, 95)),
            cuda_peak_mib=torch.cuda.max_memory_allocated(self.viewer.data.device) / 2**20)
        self.timing.content = (f"Server FPS: {self.metrics['server_fps']:.1f} · "
            f"render: {self.metrics['render_ms']:.1f} ms · JPEG+queue: {self.metrics['publish_ms']:.1f} ms")
        # print('Interaction performance: ' + json.dumps(self.metrics), flush=True)

    def close(self):
        self.closed.set()
        self._timer_thread.join()
        print(f"Interaction timings: {self.metrics}")
