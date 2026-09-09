"""Modal deployment (spec 17.5).

    modal run modal_app.py                      # full pipeline
    modal run modal_app.py::generate --n 3000   # one stage
    modal volume get physground-data /gates/contact_sheet.png .   # for Gate G2

Two things this file is careful about, both from spec 17:

*Corpus generation gets no GPU.* Rendering is a minority of per-scene cost --
physics dominates -- so hardware EGL buys about 2x end to end, while a T4 second
costs roughly 13x a core second. CPU with software rendering is both cheaper and
simpler here (spec 17.2). The GPU is for feature extraction and nothing else.

*Software rendering is chosen, never fallen into.* A CPU container that happens
to have Mesa's EGL will accept ``MUJOCO_GL=egl`` and quietly hand you llvmpipe,
with no error and every downstream wall-clock estimate invalidated (spec 17.1,
case 3). The CPU image sets ``osmesa`` explicitly and every worker runs the
preflight, which checks the renderer's identity *and* its timing.
"""

from __future__ import annotations

import modal

APP_NAME = "physground"
VOLUME_NAME = "physground-data"
DATA_DIR = "/data"

#: Never shipped to a container: version control, local environments, generated
#: artifacts (which live on the volume, not in the image), and caches.
_IGNORE = ["**/.git/**", "**/.venv/**", "**/outputs/**", "**/__pycache__/**",
           "**/.pytest_cache/**", "**/*.npz", "**/*.png"]

BASE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1", "libglew-dev",
        "libosmesa6", "libosmesa6-dev",     # software path
        "libegl1", "libgles2",              # EGL loader; only useful with a GPU
        "git",
    )
    .pip_install(
        "mujoco>=3.1", "numpy>=1.26", "scipy>=1.11", "scikit-learn>=1.3",
        "Pillow>=10.0", "matplotlib>=3.8", "PyOpenGL>=3.1",
    )
    .env({
        # One thread per worker: MuJoCo's internal threading fights
        # container-level parallelism (spec 17.4.4).
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        # The only thing pointing the pipeline at the mounted volume. No call
        # site hard-codes /data, so the same code runs locally unchanged.
        "PHYSGROUND_DATA": DATA_DIR,
        "PYTHONPATH": "/root/physground_repo:/root/physground_repo/scripts",
    })
    # modal.Mount was removed in Modal 1.0; local sources now attach to the
    # image. Left at copy=False so editing a source file does not invalidate the
    # apt and pip layers, which take minutes to rebuild.
    .add_local_dir(".", remote_path="/root/physground_repo", ignore=_IGNORE)
)

# Corpus generation. Software rendering, chosen deliberately (spec 17.2).
CPU_IMAGE = BASE.env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"})

# Feature extraction. Checkpoints are baked into the image so N containers do
# not each re-download them on cold start.
GPU_IMAGE = (
    BASE.pip_install("torch>=2.1", "torchvision>=0.16", "transformers>=4.44")
    .env({"MUJOCO_GL": "egl"})
    .run_commands(
        "python -c \"import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vitb14')\"",
        "python -c \"from transformers import VideoMAEModel, AutoImageProcessor; "
        "VideoMAEModel.from_pretrained('MCG-NJU/videomae-base'); "
        "AutoImageProcessor.from_pretrained('MCG-NJU/videomae-base')\"",
    )
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)



# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #

@app.function(image=CPU_IMAGE, cpu=1.0, memory=4096, volumes={DATA_DIR: volume},
              timeout=3600)
def generate_shard(shard_id: int, n_shards: int, n_scenes: int, master_seed: int,
                   condition: str = "base") -> dict:
    """Generate one shard of the corpus."""
    import os
    import sys
    sys.path.insert(0, "/root/physground_repo")
    sys.path.insert(0, "/root/physground_repo/scripts")

    from preflight_render import preflight
    preflight()                                    # spec 17.3: every worker, every time

    from physground import paths as P
    from physground.render import SceneRenderer
    from physground.rollout import SceneInvalid, generate_scene, scene_is_complete

    renderer = SceneRenderer(224, 224)             # one per process (spec 17.4.2)
    made = skipped = failed = 0
    try:
        for index in range(shard_id, n_scenes, n_shards):
            out_dir = P.scene_dir(condition, index)
            if scene_is_complete(out_dir, index, master_seed, condition):
                skipped += 1                       # idempotence (spec 0.6)
                continue
            try:
                generate_scene(index, master_seed, condition, out_dir, renderer)
                made += 1
            except SceneInvalid:
                failed += 1
    finally:
        renderer.close()

    # REQUIRED. Volume writes are invisible to other containers, and to
    # `modal volume get`, until committed -- a function that returns without
    # committing appears to have done nothing (spec 17.6).
    volume.commit()
    return {"shard": shard_id, "condition": condition,
            "generated": made, "skipped": skipped, "invalid": failed}


@app.function(image=GPU_IMAGE, gpu="L4", memory=16384, volumes={DATA_DIR: volume},
              timeout=3600)
def extract_shard(encoder: str, shard_id: int, n_shards: int, condition: str = "base",
                  store_patches: bool = False) -> dict:
    """Extract features for one shard."""
    import sys
    sys.path.insert(0, "/root/physground_repo")

    from physground.features import extract
    report = extract(encoder, condition, shard_id, n_shards,
                     store_patches=store_patches, batch_size=128)
    volume.commit()
    return report


@app.function(image=GPU_IMAGE, cpu=8.0, memory=32768, volumes={DATA_DIR: volume},
              timeout=7200)
def run_experiment(exp: str, seed: int) -> dict:
    """Run one experiment at one probe seed."""
    import sys
    sys.path.insert(0, "/root/physground_repo")

    from physground.experiments import EXPERIMENTS
    report = EXPERIMENTS[exp](seed=seed)
    volume.commit()
    return report


@app.function(image=CPU_IMAGE, cpu=2.0, memory=8192, volumes={DATA_DIR: volume},
              timeout=1800)
def run_gates(pilot_scenes: int = 50, corpus_scenes: int = 3000, seed: int = 0) -> dict:
    """Run G1-G3 and write the contact sheet a human must open."""
    import sys
    sys.path.insert(0, "/root/physground_repo")

    from physground import features as feature_module
    from physground import gates as gate_module
    from physground import paths as P
    from physground.figures import decorrelation_heatmap

    out_dir = P.gates_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = feature_module.scene_indices_for("base")[:pilot_scenes]
    targets = feature_module.load_targets("base", scenes)

    g1 = gate_module.gate_g1(corpus_scenes, seed, out_dir=out_dir)
    decorrelation_heatmap(g1, P.figures_dir() / "fig1_decorrelation.png")
    g3 = gate_module.gate_g3(targets, targets["scene_index"], out_dir=out_dir)
    pairs = [("base", i) for i in scenes[:20]]
    pairs += [("occluded", i) for i in feature_module.scene_indices_for("occluded")[:20]]
    g2 = gate_module.gate_g2(pairs, out_dir / "contact_sheet.png", seed=seed)

    volume.commit()
    return {"G1": {k: v for k, v in g1.items() if k not in ("rho_matrix", "pairs")},
            "G2": g2, "G3": g3}


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

@app.local_entrypoint()
def main(n_base: int = 3000, n_occluded: int = 1000, master_seed: int = 0,
         n_gen_shards: int = 20, n_extract_shards: int = 8, seeds: str = "0,1,2,3,4") -> None:
    """Run the pipeline in spec 10's order, so each stage gates the next."""
    # ~20 shards, not 100. Corpus generation is ~25 minutes of aggregate work;
    # split across 100 containers that is ~15 s of work behind a ~20 s cold
    # start, so you pay more for booting than computing (spec 17.6).
    for condition, count in (("base", n_base), ("occluded", n_occluded)):
        results = list(generate_shard.starmap(
            [(i, n_gen_shards, count, master_seed, condition) for i in range(n_gen_shards)]))
        print(f"[modal] {condition}: generated "
              f"{sum(r['generated'] for r in results)}, skipped "
              f"{sum(r['skipped'] for r in results)}, invalid "
              f"{sum(r['invalid'] for r in results)}")

    print("[modal] gates:", run_gates.remote(corpus_scenes=n_base, seed=master_seed))
    print("[modal] Gate G2 needs a human. Fetch it with:")
    print(f"[modal]   modal volume get {VOLUME_NAME} /gates/contact_sheet.png .")

    for encoder in ("dinov2_b", "random_b", "raw_pixel"):
        for condition in ("base", "occluded"):
            list(extract_shard.starmap(
                [(encoder, i, n_extract_shards, condition, False)
                 for i in range(n_extract_shards)]))
        print(f"[modal] extracted {encoder}")

    for exp in ("A", "B", "C", "D", "E"):
        for seed in (int(s) for s in seeds.split(",")):
            report = run_experiment.remote(exp, seed)
            if report.get("passed") is False:
                raise SystemExit(
                    f"[modal] Experiment {exp} seed {seed} met its kill condition: "
                    f"{report.get('kill_condition')}. Stopping (spec 10).")
            if exp in ("A", "B"):
                break        # kill-switches only need one seed to trip
        print(f"[modal] experiment {exp} done")
