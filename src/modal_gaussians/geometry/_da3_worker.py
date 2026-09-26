"""Standalone worker: run with DA3's Python environment, without importing this project."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import torch


def main(request_path):
    import depth_anything_3.api as da3_api
    from depth_anything_3.api import DepthAnything3

    # DA3 is a namespace package: its package-level __file__ is None.
    package = Path(da3_api.__file__).parent
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("DA3 preparation requires CUDA in the selected Python environment")
    torch.manual_seed(42)
    np.random.seed(42)
    model = DepthAnything3.from_pretrained(request["model"]).to("cuda").eval()
    output = Path(request["output"])
    for number, group in enumerate(request["groups"]):
        print(f"DA3 group {number+1}/{len(request['groups'])}", flush=True)
        inputs = [request["inputs"][i] for i in group["context"]]
        with torch.inference_mode():
            prediction = model.inference(
                image=[i["image"] for i in inputs],
                intrinsics=np.asarray([i["K"] for i in inputs], dtype=np.float32),
                extrinsics=np.asarray([i["world_to_camera"] for i in inputs], dtype=np.float32),
                align_to_input_ext_scale=True, infer_gs=False,
                process_res=request["process_res"], process_res_method="upper_bound_resize")
        if prediction.conf is None:
            raise ValueError("Use a DA3 multi-view model that publishes confidence")
        for target in group["targets"]:
            index = group["context"].index(target)
            np.savez(output / f"{target:06d}.npz", depth=prediction.depth[index],
                     confidence=prediction.conf[index], K=prediction.intrinsics[index],
                     extrinsics=prediction.extrinsics[index, :3])
        del prediction
    sources = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(package.rglob("*")) if p.suffix in (".py", ".yaml", ".yml")}
    runtime = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(), "numpy": np.__version__,
               "da3": importlib.metadata.version("depth-anything-3"), "sources": sources,
               "seed": 42, "align_to_input_ext_scale": True, "infer_gs": False,
               "process_res_method": "upper_bound_resize"}
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
