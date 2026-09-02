"""Local, single-editor Gradio front end for optional SAM + XMem preparation."""

from contextlib import contextmanager
from functools import partial
import gc
import json
from pathlib import Path
from threading import Event, Lock

import gradio as gr
import torch

from modal_gaussians.preparation import (
    CHECKPOINTS,
    MaskPrompt,
    PreparationCancelled,
    PreparationWorkspace,
    PreparedImages,
    XMemTracker,
    checkpoint_identity,
    inspect_images,
    read_rgb,
)


def _report_progress(progress: gr.Progress, count: int, total: int, message: str) -> None:
    """Adapt Gradio's return-valued progress call to the preparation callback."""
    progress((count, total) if total else None, desc=message)


class MaskController:
    """One mutable prompt, with a shared lock and cooperative cancellation."""

    def __init__(self, root_dir: Path, checkpoint_dir: Path, device: str):
        """Validate weights up front; load SAM/XMem only when needed."""
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; fix the environment or explicitly use --device cpu")
        self.workspace = PreparationWorkspace(root_dir)
        self.checkpoints = checkpoint_identity(checkpoint_dir)
        self.prompt = MaskPrompt(checkpoint_dir, device)
        self.lock = Lock()
        self.cancel_event = Event()
        self.images: PreparedImages | None = None
        self.sequence = ""
        self.fps = 0.0
        self.frame_index = 0

    @contextmanager
    def operation(self):
        """Reject concurrent clicks/submits rather than queueing stale actions."""
        if not self.lock.acquire(blocking=False):
            raise gr.Error("Preparation is busy. Wait or cancel the current operation.")
        self.cancel_event.clear()
        try:
            yield
        except (ValueError, RuntimeError, OSError) as error:
            raise gr.Error(str(error)) from error
        finally:
            self.lock.release()

    def cancel(self) -> str:
        """Request cancellation without competing for the model/output lock."""
        if not self.lock.locked():
            return "No active operation."
        self.cancel_event.set()
        return "Cancel requested; waiting for the current frame/model call to finish."

    def load(self, sequence: str, external: str, fps: float | None):
        """Bind the displayed prompt to a validated input sequence and its FPS."""
        images_path, _, metadata = self.workspace.paths(sequence)
        if external.strip():
            images_path = Path(external).expanduser().resolve()
        if metadata.is_file():
            record = json.loads(metadata.read_text(encoding="utf-8"))
            if record.get("images") == str(images_path) and record.get("source_video"):
                fps = float(record["fps_hz"])
        if fps is None or not 0 < float(fps) < float("inf"):
            raise ValueError("Specify the image sequence's actual FPS")
        images = inspect_images(images_path, self.cancel_event)
        self.prompt.set_image(read_rgb(images.paths[0]))
        self.images, self.sequence, self.fps = images, sequence, float(fps)
        self.frame_index = 0
        return self.prompt.preview(), gr.Slider(value=0, maximum=len(images.paths)-1), self.fps

    def select_frame(self, index: int):
        """Clear all masks/features when moving the prompting frame slider."""
        if self.images is None:
            raise ValueError("Load a sequence first")
        self.images.assert_unchanged()
        if not 0 <= int(index) < len(self.images.paths):
            raise ValueError("Prompt frame is outside the sequence")
        self.prompt.set_image(read_rgb(self.images.paths[int(index)]))
        self.frame_index = int(index)
        return self.prompt.preview()

    def track(self, progress):
        """Free SAM VRAM, run streaming XMem, and always release tracker memory."""
        if self.images is None:
            raise ValueError("Load a sequence and create a mask first")
        mask = self.prompt.mask()
        if not mask.any():
            raise ValueError("Create a nonempty foreground mask first")
        self.images.assert_unchanged()
        self.prompt.release()
        tracker = None
        try:
            tracker = XMemTracker(
                self.prompt.checkpoint_directory / CHECKPOINTS["xmem"], self.prompt.device
            )
            return self.workspace.track(
                self.images, self.sequence, fps=self.fps,
                prompt_index=self.frame_index, prompt_mask=mask, tracker=tracker,
                checkpoint_info=self.checkpoints, cancel=self.cancel_event, progress=progress,
            )
        finally:
            if tracker is not None:
                tracker.clear_memory()
            del tracker
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def build_mask_app(root_dir: Path, checkpoint_dir: Path, device: str = "cuda") -> gr.Blocks:
    """Build the local GUI without starting a server (also useful for smoke checks)."""
    controller = MaskController(root_dir, checkpoint_dir, device)
    with gr.Blocks(title="Modal Gaussians — Mask preparation", analytics_enabled=False) as app:
        gr.Markdown(
            "# Video frames → SAM mask → XMem tracking\n"
            "Local single-editor workspace. Same-name outputs are replaced after validation. "
            "Re-extraction clears that sequence's old masks. Raw videos are never modified."
        )
        gr.Markdown(f"Output root: `{controller.workspace.root}`")
        with gr.Tab("1 · Extract video (optional)"):
            video = gr.Textbox(label="Local video path (MOV / MP4)")
            extraction_name = gr.Textbox(label="Sequence name", placeholder="corn1")
            with gr.Row():
                start = gr.Number(label="Start (seconds)", value=0)
                # Number converts None to zero in Gradio; text preserves optional blanks.
                end = gr.Textbox(label="End (seconds; blank = end of video)", value="")
                output_fps = gr.Textbox(label="Output FPS (required)", value="", placeholder="e.g. 30")
                height = gr.Textbox(label="Height (blank = original)", value="")
            extract = gr.Button("Extract / replace frames", variant="primary")
        with gr.Tab("2 · Prompt and track"):
            sequence = gr.Textbox(label="Sequence name", placeholder="corn1")
            external = gr.Textbox(label="Existing PNG directory (blank = root/images/sequence)")
            fps = gr.Number(label="Sequence FPS (read from extraction metadata when available)", value=None)
            load = gr.Button("Load sequence")
            loaded = gr.Textbox(label="Loaded input", interactive=False)
            frame_index = gr.Slider(label="Prompt frame index (not the optical-flow reference)",
                                    minimum=0, maximum=1, step=1, value=0)
            features = gr.Button("Get SAM features")
            canvas = gr.Image(label="Click to select foreground/background", type="numpy",
                              interactive=False, height=600, buttons=["fullscreen", "download"])
            polarity = gr.Radio(["Foreground (+)", "Background (-)"], value="Foreground (+)",
                                label="Point type")
            with gr.Row():
                clear = gr.Button("Clear current points")
                add = gr.Button("Add foreground / start another")
                reset = gr.Button("Clear all masks")
            track = gr.Button("Track / replace masks", variant="primary")
        cancel = gr.Button("Cancel current operation")
        status = gr.Textbox(label="Status", value="Load a video or an existing PNG sequence.", interactive=False)

        def extract_video(path, name, rate, begin, finish, size, progress=gr.Progress()):
            """Publish new frames only after decoding/validation succeeds."""
            with controller.operation():
                if not rate.strip():
                    raise gr.Error("Output FPS is required")
                resolved_end = float(finish) if finish.strip() else None
                resolved_height = int(size) if size.strip() else None
                try:
                    record = controller.workspace.extract(
                        path, name, fps=float(rate), start=0 if begin is None else begin,
                        end=resolved_end, height=resolved_height,
                        cancel=controller.cancel_event,
                        progress=partial(_report_progress, progress),
                    )
                except PreparationCancelled as error:
                    return str(error), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
                controller.images = None
                controller.prompt.release()
                return (f"Extracted {record['frame_count']} frames at {record['fps_hz']:g} FPS. "
                        "Old masks cleared; load the sequence to create new masks.",
                        name, "", record["fps_hz"], None, "No sequence loaded.")

        def load_sequence(name, directory, rate):
            """Load/bind all downstream prompting actions to this input snapshot."""
            with controller.operation():
                preview, slider, resolved_fps = controller.load(name, directory, rate)
                assert controller.images is not None
                return (preview, slider, resolved_fps, "Sequence loaded; get SAM features.",
                        f"{controller.images.directory} | {len(controller.images.paths)} frames | "
                        f"{resolved_fps:g} FPS | output masks: {controller.workspace.paths(name)[1]}")

        def change_frame(index):
            """Reset prompt state on a slider release."""
            with controller.operation():
                return controller.select_frame(index), "Frame changed; get SAM features again."

        def get_features():
            """Compute SAM features for the bound prompting frame."""
            with controller.operation():
                if controller.images is None:
                    raise gr.Error("Load a sequence first")
                controller.images.assert_unchanged()
                controller.prompt.features()
                return "Features ready. Add positive/negative points on the image."

        def select_point(mode, event: gr.SelectData):
            """Use a concrete Gradio event annotation for image click coordinates."""
            with controller.operation():
                x, y = event.index
                return controller.prompt.add_point(int(x), int(y), mode == "Foreground (+)")

        def clear_points():
            """Reset only the currently edited foreground."""
            with controller.operation():
                controller.prompt.clear_points()
                return controller.prompt.preview()

        def add_foreground():
            """Keep this foreground in the union while prompting another."""
            with controller.operation():
                controller.prompt.add_foreground()
                return controller.prompt.preview()

        def clear_all():
            """Reload the current image and discard all masks and embeddings."""
            with controller.operation():
                return controller.select_frame(controller.frame_index), "Masks cleared; get SAM features."

        def track_masks(progress=gr.Progress()):
            """Run the single streaming job; cancellation leaves old results intact."""
            with controller.operation():
                try:
                    record = controller.track(partial(_report_progress, progress))
                except PreparationCancelled as error:
                    return str(error)
                return f"Ready: {record['frame_count']} masks saved to {record['masks']}. SAM released."

        # Non-queued edits fail immediately while a long job holds the shared lock.
        # Gradio attaches these event methods dynamically through its metaclass.
        getattr(extract, "click")(extract_video, [video, extraction_name, output_fps, start, end, height],
                      [status, sequence, external, fps, canvas, loaded], concurrency_limit=None)
        getattr(load, "click")(load_sequence, [sequence, external, fps], [canvas, frame_index, fps, status, loaded],
                   queue=False)
        getattr(frame_index, "release")(change_frame, frame_index, [canvas, status], queue=False)
        getattr(features, "click")(get_features, outputs=status, concurrency_limit=None)
        getattr(canvas, "select")(select_point, polarity, canvas, queue=False)
        getattr(clear, "click")(clear_points, outputs=canvas, queue=False)
        getattr(add, "click")(add_foreground, outputs=canvas, queue=False)
        getattr(reset, "click")(clear_all, outputs=[canvas, status], queue=False)
        getattr(track, "click")(track_masks, outputs=status, concurrency_limit=None)
        getattr(cancel, "click")(controller.cancel, outputs=status, queue=False)
    return app.queue()


def run_mask_gui(*, root_dir: Path, checkpoint_dir: Path, port: int = 8890,
                 device: str = "cuda") -> None:
    """Start only a loopback server; never enable sharing or upload raw videos."""
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    app = build_mask_app(root_dir, checkpoint_dir, device)
    app.launch(server_name="127.0.0.1", server_port=port, share=False, inbrowser=False)
