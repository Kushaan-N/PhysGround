#!/usr/bin/env python
"""Extract and cache frozen features (spec 7).

    python scripts/extract_features.py --encoder dinov2_b --shard 0 --n-shards 8
    python scripts/extract_features.py --encoder vjepa2 --condition base --no-patches

Idempotent: a shard already complete under the current config is skipped, so a
preempted array job re-runs only what it lost (spec 0.6).
"""

from __future__ import annotations

import argparse
import json

# No MUJOCO_GL here on purpose. Extraction reads PNGs back through
# physground.frames, which imports no mujoco, so this stage needs no GL backend
# at all -- and hard-coding one would make it fail on a machine whose valid
# backends differ, for a reason unrelated to what it is doing.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", required=True,
                        help="dinov2_b | random_b | raw_pixel | videomae_b | vjepa2")
    parser.add_argument("--condition", default="base", choices=("base", "occluded"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", default="float16", choices=("float16", "float32"),
                        help="cached precision; fp16 is settled by Gate G4 (spec 3.6)")
    parser.add_argument("--no-patches", action="store_true",
                        help="skip patch tokens (~31 GB for dinov2_b; spec 7.3)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=0,
                        help="passed to encoders whose weights are sampled (random_b)")
    args = parser.parse_args()

    import numpy as np    # noqa: E402
    import torch          # noqa: E402
    import random         # noqa: E402

    # Spec 0.5: seed every generator, even where extraction is deterministic --
    # random_b's weights are drawn at construction.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from physground.features import extract   # noqa: E402

    encoder_kwargs = {}
    if args.device:
        encoder_kwargs["device"] = args.device
    if args.encoder == "random_b":
        encoder_kwargs["seed"] = args.seed

    report = extract(args.encoder, args.condition, args.shard, args.n_shards,
                     batch_size=args.batch_size, dtype=args.dtype,
                     store_patches=not args.no_patches, force=args.force,
                     encoder_kwargs=encoder_kwargs)
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
