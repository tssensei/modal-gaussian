"""Synthetic selection -> flow -> FFT export -> modal binding; no real experiments."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from modal_gaussians.cli import build_parser
from modal_gaussians.flow import reference_selection as rs
from modal_gaussians.flow.sea_raft import compute_flow
from modal_gaussians.flow.storage import open_array
from modal_gaussians.iteration_cache import identity
from modal_gaussians.motion.neural.modal_similarity_artifact import _modal_view
from modal_gaussians.spectrum_cache import build_spectrum, save_selection, export_selection
from modal_gaussians import rendered_design


class ReferenceSelectionTest(unittest.TestCase):
    def test_selection_to_modal_with_legacy_and_mismatch_guards(self):
        cv2.setNumThreads(1)
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            images, masks, metadata = [root / name for name in ("images", "masks", "geometry")]
            for path in (images, masks, metadata):
                path.mkdir()
            fixed = np.zeros((32, 48), bool)
            fixed[8:24, 15:25] = True
            fixed[12:17, 25:36] = True
            names = ["00001", "00002", "00003"]
            frames = []
            for name, shift in zip(names, (5, 2, 0)):
                mask = np.roll(fixed, shift, axis=1)
                rgb = np.full((32, 48, 3), 60, np.uint8)
                rgb[mask] = [50, 180, 60]
                frames.append(rgb)
                cv2.imwrite(str(images / f"{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(masks / f"{name}.png"), np.uint8(mask)*255)
            source = {"format": "modal_gaussians.flow_analysis", "version": 7,
                "inputs": {"sequence": {"image_directory": str(images), "mask_directory": str(masks)}},
                "frame_names": names, "fps_hz": 20, "reference_frame_index": 0,
                "reference_frame_name": names[0], "arrays": {"flow": {"shape": [3,32,48,2]}}}
            (metadata / "manifest.json").write_text(json.dumps(source))
            camera = NS(role="reference", label="view1", height=32, width=48,
                        to_manifest_record=lambda: {"camera_identity": "camera"})
            camera.to = lambda _: camera
            scene = NS(manifest={"static_scene_identity": "scene", "foreground_identity": "fg"},
                requires_grad_=lambda _: None,
                render_batch=lambda *a, **kw: ({"rgb": torch.from_numpy(frames[-1][None]/255.),
                    "foreground_mask": torch.from_numpy(fixed[None].astype(np.float32))}, {}))
            scene.eval = lambda: scene
            with patch("modal_gaussians.static.load_static_scene", return_value=scene), \
                 patch("modal_gaussians.static.cameras_from_scene_manifest", return_value=[camera]):
                selection = rs.select_reference(scene_dir=root, label="view1", reuse_stabilization=metadata,
                    output_dir=root / "selection", target_mask="green", workers=2)
                with self.assertRaises(FileExistsError):
                    rs.select_reference(scene_dir=root, label="view1", reuse_stabilization=metadata,
                        output_dir=selection)
            selected = json.loads((selection / "manifest.json").read_text())
            self.assertEqual(selected["selected"]["index"], 2)
            self.assertEqual(selected["selected"]["boundary_mean_px"], 0)
            self.assertEqual(selected["frames_scored"], 3)
            self.assertTrue((selection / "comparison_subject.png").is_file())
            with self.assertRaisesRegex(ValueError, "Empty"):
                rs.score_mask(np.zeros_like(fixed), rs.target_data(fixed))
            near = np.roll(fixed, 2, axis=1)
            np.testing.assert_allclose(rs.score_mask(near, rs.target_data(fixed))["boundary_mean_px"],
                rs.score_mask(fixed, rs.target_data(near))["boundary_mean_px"])

            references = []
            class Model:
                args = NS(iters=1)
                def __call__(self, reference, current, **kwargs):
                    references.append(reference.numpy().copy())
                    return {"final": torch.ones(1, 2, 32, 48)}
            with patch("torch.cuda.is_available", return_value=True), \
                 patch.object(torch.Tensor, "cuda", lambda self: self), \
                 patch("modal_gaussians.flow.sea_raft.load_model", return_value=Model()):
                new_flow = compute_flow(images=images, reuse_stabilization=metadata,
                    reference_selection=selection, output_dir=root/"flow", sea_raft_repo=root, model_dir=root)
                legacy_flow = compute_flow(images=images, reuse_stabilization=metadata,
                    output_dir=root/"legacy_flow", sea_raft_repo=root, model_dir=root)
            expected = np.moveaxis(frames[-1], -1, 0)[None]
            np.testing.assert_array_equal(references[0], expected)
            values = open_array(new_flow/"flow.zarr")
            np.testing.assert_array_equal(values[2], np.zeros((32,48,2)))
            np.testing.assert_array_equal(values[0], np.ones((32,48,2)))
            values.store.close()
            old = json.loads((legacy_flow/"manifest.json").read_text())
            self.assertNotIn("reference_selection", old)
            self.assertEqual(old["reference_frame_index"], 0)
            region = root/"region.npy"
            np.save(region, fixed)
            cache = build_spectrum(views=[("view1",new_flow)], fft_length=4, output_dir=root/"fft",
                                   region_paths=[("view1",region)])
            selection_file = save_selection(cache, [1], root/"bins.json")
            exports = export_selection(cache, selection_file, root/"exports")
            export = exports/"bin_0001/view1"
            exported = json.loads((export/"manifest.json").read_text())
            self.assertEqual(exported["reference_selection"]["identity"], selected["selection_identity"])
            flow_record = {"path": str(metadata), "identity": "original-flow", "manifest": source}
            view = {"label": "view1", "camera_identity": "camera", "shape_hw": [32,48]}
            field, mask, _ = _modal_view(export, flow_record, view, 5, static_scene_identity="scene")
            self.assertEqual(field.shape, (32,48,2))
            np.testing.assert_array_equal(mask, fixed)
            field._mmap.close()
            with self.assertRaisesRegex(ValueError, "static geometry"):
                _modal_view(export, flow_record, view, 5, static_scene_identity="other")
            with self.assertRaisesRegex(ValueError, "camera/view"):
                _modal_view(export, flow_record, {**view, "camera_identity":"other"}, 5)
            changed = copy.deepcopy(exported)
            changed.pop("reference_selection")
            with self.assertRaisesRegex(ValueError, "requires"):
                rs.motion_reference(changed, source)
            changed = copy.deepcopy(exported)
            changed["reference_selection"]["contract"]["reference_frame_index"] = 1
            with self.assertRaisesRegex(ValueError, "contract"):
                rs.motion_reference(changed, source)
            with self.assertRaisesRegex(ValueError, "contract"):
                rs.motion_reference(exported, {**source, "fps_hz": 30})
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                rs.reference_binding({**selected, "status":"running"})

            # Preview retains geometric provenance while exposing the motion reference.
            completed_view = {"index":0, **view, "flow_identity":"original-flow",
                "motion_reference":{"reference_frame_name":names[2], "reference_frame_index":2,
                                    "selection_identity":selected["selection_identity"]}}
            completed = NS(path=root, manifest={"static_scene_identity":"scene",
                "foreground_identity":"fg", "views":[completed_view]})
            flow_stub = NS(manifest=source, arrays=NS(mask_union=fixed, flow=np.empty((3,32,48,2))))
            camera.name = "reference/view1"
            with patch.object(rendered_design, "load_static_scene", return_value=scene), \
                 patch.object(rendered_design, "cameras_from_scene_manifest", return_value=[camera]), \
                 patch.object(rendered_design, "flow_artifact_identity", return_value="original-flow"):
                loaded = rendered_design._load_sources(scene_dir=root, completed_modes_dir=root,
                    views=[NS(label="view1",flow_artifact=metadata)], device="cpu",
                    validated_completed=completed, flow_loader=lambda _:flow_stub)
            self.assertEqual(loaded[-1][0]["flow_reference_frame_index"], 0)
            self.assertEqual(loaded[-1][0]["motion_reference"], completed_view["motion_reference"])

            # Changing the motion reference must keep the existing stabilized grid.
            stable = metadata/"stabilized"
            stable.mkdir()
            (stable/"manifest.json").write_text(json.dumps({"frames":names, "fps_hz":20,
                                                           "reference_frame":names[0]}))
            source["stabilized_sequence"] = {"path":"stabilized"}
            (metadata/"manifest.json").write_text(json.dumps(source))
            seq = rs.sequence_metadata(metadata, images)
            self.assertEqual(seq["image_dir"], stable/"images")
            self.assertEqual(seq["mask_dir"], stable/"masks")
            self.assertEqual(seq["reference"], 0)
            with self.assertRaisesRegex(ValueError, "contract"):
                rs.motion_reference(exported, source)

            args = build_parser().parse_args(["flow","select-reference","--scene",str(root),
                "--view-label","view1","--reuse-stabilization",str(metadata),"--output",str(root/"new")])
            self.assertEqual(args.target_mask, "alpha")


if __name__ == "__main__":
    unittest.main()
