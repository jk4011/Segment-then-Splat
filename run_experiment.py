"""
Master orchestration script for Segment-then-Splat experiments.
Runs the full pipeline on LeRF, 3D-OVS, DL3DV datasets with timing and evaluation.
Uses 2 GPUs in parallel (scene-level parallelism).
"""
import os
import sys
import json
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
AUTOSEG_DIR = os.path.join(PROJECT_ROOT, "third_party", "AutoSeg-SAM2")
PYTHON = "/opt/conda/envs/segment_then_splat/bin/python"

DATASETS = {
    "lerf": {
        "root": "/root/data1/jinhyeok/seg123/dataset/lerf_ovs",
        "scenes": ["figurines", "ramen", "teatime", "waldo_kitchen"],
        "label_format": "lerf",
        "label_root": "/root/data1/jinhyeok/seg123/dataset/lerf_ovs/label",
        "image_ext": ".jpg",
    },
    "3d_ovs": {
        "root": "/root/data1/jinhyeok/seg123/dataset/3d_ovs",
        "scenes": ["bed", "bench", "blue_sofa", "covered_desk", "lawn", "office_desk", "room", "snacks", "sofa", "table"],
        "label_format": "3d_ovs",
        "image_ext": ".jpg",
    },
    "dl3dv": {
        "root": "/root/data1/jinhyeok/seg123/dataset/dl3dv",
        "scenes": ["furniture_shop", "office", "park_bench_car", "road_car_building"],
        "label_format": "lerf",
        "label_root": "/root/data1/jinhyeok/seg123/dataset/dl3dv/label",
        "image_ext": ".png",
    },
}


def get_scene_dir(dataset_name, scene_name):
    return os.path.join(DATASETS[dataset_name]["root"], scene_name)


def get_label_dir(dataset_name, scene_name):
    ds = DATASETS[dataset_name]
    if ds["label_format"] == "3d_ovs":
        return os.path.join(ds["root"], scene_name, "segmentations")
    else:
        return os.path.join(ds["label_root"], scene_name, "gt")


def get_output_dir(scene_name):
    return os.path.join(PROJECT_ROOT, "output", scene_name)


def get_autoseg_output_dir(scene_name):
    return os.path.join(AUTOSEG_DIR, "output", scene_name)


def run_cmd(cmd, env=None, cwd=None):
    """Run a command and return (returncode, stdout, stderr)."""
    print(f"[CMD] {cmd}")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    result = subprocess.run(cmd, shell=True, env=merged_env, cwd=cwd,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed (rc={result.returncode}):\n{result.stdout[-2000:]}")
    return result.returncode, result.stdout


def create_train_test_split(dataset_name, scene_name):
    """Create train.txt and test.txt for a scene."""
    scene_dir = get_scene_dir(dataset_name, scene_name)
    ds = DATASETS[dataset_name]
    image_dir = os.path.join(scene_dir, "images")
    all_images = sorted(os.listdir(image_dir))

    if ds["label_format"] == "3d_ovs":
        seg_dir = os.path.join(scene_dir, "segmentations")
        test_frames = set()
        for d in os.listdir(seg_dir):
            dp = os.path.join(seg_dir, d)
            if os.path.isdir(dp):
                test_frames.add(d + ds["image_ext"])
        test_list = sorted(test_frames & set(all_images))
    else:
        label_dir = os.path.join(ds["label_root"], scene_name, "gt")
        test_frames = set()
        if os.path.exists(label_dir):
            for d in os.listdir(label_dir):
                if os.path.isdir(os.path.join(label_dir, d)):
                    test_frames.add(d + ds["image_ext"])
        test_list = sorted(test_frames & set(all_images))

    train_list = all_images

    train_path = os.path.join(scene_dir, "train.txt")
    test_path = os.path.join(scene_dir, "test.txt")
    with open(train_path, "w") as f:
        f.write("\n".join(train_list) + "\n")
    with open(test_path, "w") as f:
        f.write("\n".join(test_list) + "\n")
    print(f"  Created train.txt ({len(train_list)} images) and test.txt ({len(test_list)} images)")


def step1_autoseg(dataset_name, scene_name, gpu_id):
    """Run AutoSeg-SAM2 for all 3 levels. Returns SAM inference time."""
    scene_dir = get_scene_dir(dataset_name, scene_name)
    autoseg_out = get_autoseg_output_dir(scene_name)
    os.makedirs(autoseg_out, exist_ok=True)
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}

    total_sam_time = 0.0
    for level in ["large", "middle", "small"]:
        cmd = (
            f"{PYTHON} auto-mask-fast.py "
            f"--video_path {scene_dir}/images "
            f"--output_dir {autoseg_out} "
            f"--level {level} "
            f"--detect_stride 10 --batch_size 20"
        )
        rc, _ = run_cmd(cmd, env=env, cwd=AUTOSEG_DIR)
        if rc != 0:
            print(f"  [WARN] AutoSeg failed for {scene_name}/{level}")
            continue

        timing_file = os.path.join(autoseg_out, f"sam_inference_time_{level}.json")
        if os.path.exists(timing_file):
            with open(timing_file) as f:
                total_sam_time += json.load(f).get("sam_inference_sec", 0)

    return total_sam_time


def step2_preprocess_mask(dataset_name, scene_name):
    """Run preprocess_mask.py. Returns elapsed time."""
    scene_dir = get_scene_dir(dataset_name, scene_name)
    autoseg_out = get_autoseg_output_dir(scene_name)
    t0 = time.time()
    cmd = (
        f"{PYTHON} {PROJECT_ROOT}/helpers/preprocess_mask.py "
        f"--mask_root {autoseg_out} "
        f"--out_root {scene_dir}/ "
        f"--image_path {scene_dir}/images"
    )
    run_cmd(cmd)
    return time.time() - t0


def step4_obj_init(dataset_name, scene_name):
    """Run object_specific_initialization.py. Returns elapsed time."""
    scene_dir = get_scene_dir(dataset_name, scene_name)
    t0 = time.time()
    cmd = f"{PYTHON} {PROJECT_ROOT}/helpers/object_specific_initialization.py --scene_root {scene_dir}/"
    run_cmd(cmd)
    return time.time() - t0


def step5_train(dataset_name, scene_name, gpu_id):
    """Run train.py. Returns (base_time, obj_time) from timing.json."""
    scene_dir = get_scene_dir(dataset_name, scene_name)
    output_dir = get_output_dir(scene_name)
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    cmd = (
        f"{PYTHON} {PROJECT_ROOT}/train.py "
        f"-s {scene_dir}/ -m {output_dir} "
        f"--eval --iterations 40000 --num_sample_objects 3 "
        f"--densify_until_iter 20000 --partial_mask_iou 0.3"
    )
    run_cmd(cmd, env=env)

    timing_file = os.path.join(output_dir, "timing.json")
    if os.path.exists(timing_file):
        with open(timing_file) as f:
            data = json.load(f)
        return data.get("base_render_loss_sec", 0), data.get("obj_render_loss_sec", 0)
    return 0, 0


def step6_render_and_eval(dataset_name, scene_name, gpu_id):
    """Run render_objs.py + evaluation.py. Returns (mIoU, mACC, clip_time)."""
    scene_dir = get_scene_dir(dataset_name, scene_name)
    output_dir = get_output_dir(scene_name)
    label_dir = get_label_dir(dataset_name, scene_name)
    ds = DATASETS[dataset_name]
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}

    cmd = (
        f"{PYTHON} {PROJECT_ROOT}/render_objs.py "
        f"-m {output_dir}/ --mode render --skip_train"
    )
    run_cmd(cmd, env=env)

    eval_json = os.path.join(output_dir, "eval_results.json")
    cmd = (
        f"{PYTHON} {PROJECT_ROOT}/helpers/evaluation.py "
        f"--scene {scene_dir}/ --render_dir {output_dir}/ "
        f"--label_dir {label_dir} "
        f"--label_format {ds['label_format']} "
        f"--output_json {eval_json}"
    )
    run_cmd(cmd, env=env)

    if os.path.exists(eval_json):
        with open(eval_json) as f:
            data = json.load(f)
        return data.get("mIoU", 0), data.get("mACC", 0), data.get("clip_inference_sec", 0)
    return 0, 0, 0


def run_scene(dataset_name, scene_name, gpu_id):
    """Run full pipeline for a single scene on a specific GPU."""
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}/{scene_name} on GPU {gpu_id}")
    print(f"{'='*60}")

    scene_result = {"scene": scene_name, "dataset": dataset_name}

    print(f"  Step 0: Creating train/test split...")
    create_train_test_split(dataset_name, scene_name)

    print(f"  Step 1: AutoSeg-SAM2...")
    sam_time = step1_autoseg(dataset_name, scene_name, gpu_id)
    scene_result["sam_inference_sec"] = sam_time

    print(f"  Step 2: Preprocess masks...")
    preprocess_time = step2_preprocess_mask(dataset_name, scene_name)
    scene_result["preprocess_mask_sec"] = preprocess_time

    print(f"  Step 4: Object-specific initialization...")
    obj_init_time = step4_obj_init(dataset_name, scene_name)
    scene_result["obj_init_sec"] = obj_init_time

    print(f"  Step 5: Training (40k iterations)...")
    base_time, obj_time = step5_train(dataset_name, scene_name, gpu_id)
    scene_result["3d_recon_sec"] = base_time
    scene_result["3d_lifting_train_sec"] = obj_time

    print(f"  Step 6: Rendering + Evaluation...")
    miou, macc, clip_time = step6_render_and_eval(dataset_name, scene_name, gpu_id)
    scene_result["mIoU"] = miou
    scene_result["mACC"] = macc
    scene_result["clip_inference_sec"] = clip_time

    scene_result["3d_lifting_sec"] = obj_init_time + obj_time
    scene_result["etc_sec"] = preprocess_time + clip_time
    scene_result["total_sec"] = sam_time + base_time + scene_result["3d_lifting_sec"] + scene_result["etc_sec"]

    scene_json = os.path.join(get_output_dir(scene_name), "scene_result.json")
    os.makedirs(os.path.dirname(scene_json), exist_ok=True)
    with open(scene_json, "w") as f:
        json.dump(scene_result, f, indent=2)

    print(f"  Done: mIoU={miou:.4f}, mACC={macc:.4f}, SAM={sam_time:.0f}s, Recon={base_time:.0f}s, Lifting={scene_result['3d_lifting_sec']:.0f}s, Etc={scene_result['etc_sec']:.0f}s")
    return scene_result


def generate_markdown(all_results):
    """Generate Experiment.md from all results."""
    lines = ["# Experiment Results\n"]

    for ds_name, ds_label in [("lerf", "LeRF-OVS"), ("3d_ovs", "3D-OVS"), ("dl3dv", "DL3DV")]:
        ds_results = [r for r in all_results if r["dataset"] == ds_name]
        if not ds_results:
            continue

        lines.append(f"\n## {ds_label}\n")
        lines.append("| Scene | mIoU | mACC | SAM inf. | 3D Recon. | 3D Lifting | Etc | Total |")
        lines.append("|-------|------|------|----------|-----------|------------|-----|-------|")

        sums = {"mIoU": 0, "mACC": 0, "sam": 0, "recon": 0, "lift": 0, "etc": 0, "total": 0}
        for r in ds_results:
            lines.append(
                f"| {r['scene']} "
                f"| {r['mIoU']:.4f} "
                f"| {r['mACC']:.4f} "
                f"| {r['sam_inference_sec']:.0f} "
                f"| {r['3d_recon_sec']:.0f} "
                f"| {r['3d_lifting_sec']:.0f} "
                f"| {r['etc_sec']:.0f} "
                f"| {r['total_sec']:.0f} |"
            )
            sums["mIoU"] += r["mIoU"]
            sums["mACC"] += r["mACC"]
            sums["sam"] += r["sam_inference_sec"]
            sums["recon"] += r["3d_recon_sec"]
            sums["lift"] += r["3d_lifting_sec"]
            sums["etc"] += r["etc_sec"]
            sums["total"] += r["total_sec"]

        n = len(ds_results)
        lines.append(
            f"| **Mean** "
            f"| **{sums['mIoU']/n:.4f}** "
            f"| **{sums['mACC']/n:.4f}** "
            f"| **{sums['sam']/n:.0f}** "
            f"| **{sums['recon']/n:.0f}** "
            f"| **{sums['lift']/n:.0f}** "
            f"| **{sums['etc']/n:.0f}** "
            f"| **{sums['total']/n:.0f}** |"
        )

    lines.append("\n\nAll times in seconds. 3D Lifting = obj_init + train per-object portion. Etc = preprocess_mask + CLIP inference.\n")

    md_path = os.path.join(PROJECT_ROOT, "Experiment.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nExperiment.md saved to {md_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated GPU IDs")
    parser.add_argument("--datasets", type=str, default="lerf,3d_ovs,dl3dv", help="Comma-separated dataset names")
    parser.add_argument("--max_parallel", type=int, default=2, help="Max parallel scenes")
    parser.add_argument("--scene", type=str, default=None, help="Run single scene (format: dataset/scene)")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    dataset_names = [d.strip() for d in args.datasets.split(",")]

    if args.scene:
        ds, sc = args.scene.split("/")
        result = run_scene(ds, sc, gpu_ids[0])
        generate_markdown([result])
        return

    scene_queue = []
    for ds_name in dataset_names:
        if ds_name not in DATASETS:
            print(f"Unknown dataset: {ds_name}")
            continue
        for scene_name in DATASETS[ds_name]["scenes"]:
            scene_queue.append((ds_name, scene_name))

    all_results = []
    with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {}
        for i, (ds_name, scene_name) in enumerate(scene_queue):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            future = executor.submit(run_scene, ds_name, scene_name, gpu_id)
            futures[future] = (ds_name, scene_name)

        for future in as_completed(futures):
            ds_name, scene_name = futures[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"[ERROR] {ds_name}/{scene_name} failed: {e}")
                all_results.append({
                    "scene": scene_name, "dataset": ds_name,
                    "mIoU": 0, "mACC": 0,
                    "sam_inference_sec": 0, "3d_recon_sec": 0,
                    "3d_lifting_sec": 0, "etc_sec": 0, "total_sec": 0,
                })

    dataset_order = {"lerf": 0, "3d_ovs": 1, "dl3dv": 2}
    all_results.sort(key=lambda r: (dataset_order.get(r["dataset"], 99), r["scene"]))

    results_json = os.path.join(PROJECT_ROOT, "output", "all_results.json")
    os.makedirs(os.path.dirname(results_json), exist_ok=True)
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {results_json}")

    generate_markdown(all_results)


if __name__ == "__main__":
    main()
