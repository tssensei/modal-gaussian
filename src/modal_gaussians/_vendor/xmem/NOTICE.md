# XMem inference provenance

Inference sources copied from the accepted project's preproc/tracker snapshot,
originally XMem (https://github.com/hkchengrex/XMem) via Track-Anything
(https://github.com/gaomingqi/Track-Anything). Original comments are retained.
CBAM originates from https://github.com/Jongchan/attention-module.
See the accompanying MIT licenses. The modified ResNet retains torchvision's
BSD notice in LICENSE-torchvision. SAM is installed separately under Apache-2.0.

Local adaptations: internal imports use modal_gaussians._vendor.xmem;
ResNet pretrained downloads are disabled because the complete XMem checkpoint
is loaded strictly afterwards; checkpoint loading explicitly uses weights_only.
Memory matching is locally chunked along the query-pixel axis (256 feature
positions per chunk by default). Each chunk matches against the complete memory
bank with the original group-specific top-k/softmax; usage is accumulated and
memory age advances once per frame. Resolution, precision, memory retention and
consolidation rules are unchanged. FP32 reduction order may cause small numerical
differences. Setting query_chunk_size=0 uses a single dense query for validation.
Training losses, demos, visualization and SAM-refinement helpers are not vendored.
The small binary inference wrapper and GUI live in preparation.py/mask_gui.py.
