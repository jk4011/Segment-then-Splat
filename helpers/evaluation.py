import torch
import clip
from PIL import Image
import os
import json
import time
from tqdm import tqdm
from torchvision import transforms
import argparse

device = "cuda"

def load_images_masks(scene_path):

    image_dir = os.path.join(scene_path, "images")
    default_mask_dir = os.path.join(scene_path, "multiview_masks_default_merged")
    train_txt = os.path.join(scene_path, "train.txt")
    with open(train_txt, "r") as f:
        train_list = f.readlines()
    train_list = [x.strip() for x in train_list]
    image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f in train_list]

    num_large_objects = len(os.listdir(default_mask_dir))
    large_object_images = [[] for _ in range(num_large_objects)]
    for object_idx, object_mask_dir in tqdm(enumerate(sorted(os.listdir(default_mask_dir))), total=num_large_objects, desc="Loading large object images"):
        object_mask_path = os.path.join(default_mask_dir, object_mask_dir)
        object_masks = [os.path.join(object_mask_path, f) for f in sorted(os.listdir(object_mask_path)) if f in train_list]
        for i in range(len(image_paths)):
            image = Image.open(image_paths[i])
            object_mask = Image.open(object_masks[i])

            masked_image = Image.composite(image, Image.new("RGBA", image.size, (0, 0, 0, 255)), object_mask)
            bbox = object_mask.getbbox()
            masked_image = masked_image.crop(bbox)
            large_object_images[object_idx].append(masked_image)

    middle_mask_dir = os.path.join(scene_path, "multiview_masks_middle_merged")
    num_middle_objects = len(os.listdir(middle_mask_dir))
    middle_object_images = [[] for _ in range(num_middle_objects)]
    for object_idx, object_mask_dir in tqdm(enumerate(sorted(os.listdir(middle_mask_dir))), total=num_middle_objects, desc="Loading middle object images"):
        object_mask_path = os.path.join(middle_mask_dir, object_mask_dir)
        object_masks = [os.path.join(object_mask_path, f) for f in sorted(os.listdir(object_mask_path)) if f in train_list]
        for i in range(len(image_paths)):
            image = Image.open(image_paths[i])
            object_mask = Image.open(object_masks[i])

            masked_image = Image.composite(image, Image.new("RGBA", image.size, (0, 0, 0, 255)), object_mask)
            bbox = object_mask.getbbox()
            masked_image = masked_image.crop(bbox)
            middle_object_images[object_idx].append(masked_image)

    small_mask_dir = os.path.join(scene_path, "multiview_masks_small_merged")
    num_small_objects = len(os.listdir(small_mask_dir))
    small_object_images = [[] for _ in range(num_small_objects)]
    for object_idx, object_mask_dir in tqdm(enumerate(sorted(os.listdir(small_mask_dir))), total=num_small_objects, desc="Loading small object images"):
        object_mask_path = os.path.join(small_mask_dir, object_mask_dir)
        object_masks = [os.path.join(object_mask_path, f) for f in sorted(os.listdir(object_mask_path)) if f in train_list]
        for i in range(len(image_paths)):
            image = Image.open(image_paths[i])
            object_mask = Image.open(object_masks[i])

            masked_image = Image.composite(image, Image.new("RGBA", image.size, (0, 0, 0, 255)), object_mask)
            bbox = object_mask.getbbox()
            masked_image = masked_image.crop(bbox)
            small_object_images[object_idx].append(masked_image)

    return large_object_images, middle_object_images, small_object_images

def extract_CLIP_embeddings(model, preprocess, large_object_images, middle_object_images, small_object_images):
    large_object_features = []
    for object_idx in tqdm(range(len(large_object_images)), desc="Extracting large object features"):
        object_features = []
        for object_img in large_object_images[object_idx]:
            image = preprocess(object_img).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
            object_features.append(image_features)
        large_object_features.append(torch.stack(object_features).mean(dim=0))
    mean_large_object_features = torch.stack(large_object_features)

    middle_object_features = []
    for object_idx in tqdm(range(len(middle_object_images)), desc="Extracting middle object features"):
        object_features = []
        for object_img in middle_object_images[object_idx]:
            image = preprocess(object_img).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
            object_features.append(image_features)
        middle_object_features.append(torch.stack(object_features).mean(dim=0))
    mean_middle_object_features = torch.stack(middle_object_features)

    small_object_features = []
    for object_idx in tqdm(range(len(small_object_images)), desc="Extracting small object features"):
        object_features = []
        for object_img in small_object_images[object_idx]:
            image = preprocess(object_img).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
            object_features.append(image_features)
        small_object_features.append(torch.stack(object_features).mean(dim=0))
    mean_small_object_features = torch.stack(small_object_features) if small_object_features else torch.empty(0)

    return mean_large_object_features, mean_middle_object_features, mean_small_object_features

def load_results(results_dir):
    transform = transforms.ToTensor()
    large_object_result_dirs = [os.path.join(results_dir, f, 'ours_40000/renders') for f in sorted(os.listdir(results_dir)) if f.startswith("test_default")]
    predicted_large_objects = []
    for object_result_dir in tqdm(large_object_result_dirs, desc="Loading rendered large objects"):
        object_images = [transform(Image.open(os.path.join(object_result_dir, f))) for f in sorted(os.listdir(object_result_dir))]
        predicted_large_objects.append(object_images)

    middle_object_result_dirs = [os.path.join(results_dir, f, 'ours_40000/renders') for f in sorted(os.listdir(results_dir)) if f.startswith("test_middle")]
    predicted_middle_objects = []
    for object_result_dir in tqdm(middle_object_result_dirs, desc="Loading rendered middle objects"):
        object_images = [transform(Image.open(os.path.join(object_result_dir, f))) for f in sorted(os.listdir(object_result_dir))]
        predicted_middle_objects.append(object_images)

    small_object_result_dirs = [os.path.join(results_dir, f, 'ours_40000/renders') for f in sorted(os.listdir(results_dir)) if f.startswith("test_small")]
    predicted_small_objects = []
    for object_result_dir in tqdm(small_object_result_dirs, desc="Loading rendered small objects"):
        object_images = [transform(Image.open(os.path.join(object_result_dir, f))) for f in sorted(os.listdir(object_result_dir))]
        predicted_small_objects.append(object_images)

    return predicted_large_objects, predicted_middle_objects, predicted_small_objects


def load_gt_labels(label_path, label_format="lerf"):
    """Load GT labels, returning list of (frame_dir_name, [(query_name, mask_tensor), ...])"""
    transform = transforms.ToTensor()
    gt_data = []

    if label_format == "3d_ovs":
        gt_dirs = sorted(os.listdir(label_path))
        for gt_dir in gt_dirs:
            frame_path = os.path.join(label_path, gt_dir)
            if not os.path.isdir(frame_path):
                continue
            mask_files = sorted(os.listdir(frame_path))
            items = []
            for mf in mask_files:
                query_name = os.path.splitext(mf)[0]
                mask = transform(Image.open(os.path.join(frame_path, mf)))
                items.append((query_name, mask))
            if items:
                gt_data.append((gt_dir, items))
    else:
        gt_dirs = sorted(os.listdir(label_path))
        for gt_dir in gt_dirs:
            frame_path = os.path.join(label_path, gt_dir)
            if not os.path.isdir(frame_path):
                continue
            mask_files = sorted(os.listdir(frame_path))
            items = []
            for mf in mask_files:
                query_name = os.path.splitext(mf)[0]
                mask = transform(Image.open(os.path.join(frame_path, mf)))
                items.append((query_name, mask))
            if items:
                gt_data.append((gt_dir, items))

    return gt_data


def calculate_IoU(gt_data, mean_large_object_features, mean_middle_object_features, mean_small_object_features,
                  predicted_large_objects, predicted_middle_objects, predicted_small_objects, model):
    all_ious = []
    all_accs = []

    for gt_idx, (gt_dir, items) in enumerate(gt_data):
        queries = [item[0] for item in items]
        gt_masks = [item[1] for item in items]

        text_queries = clip.tokenize(queries).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_queries)

        large_similarity = (mean_large_object_features @ text_features.T).squeeze(1)
        middle_similarity = (mean_middle_object_features @ text_features.T).squeeze(1)

        topk_value_large, topk_indice_large = torch.topk(large_similarity, 1, largest=True, dim=0)
        topk_value_middle, topk_indice_middle = torch.topk(middle_similarity, 1, largest=True, dim=0)

        ious = []
        accs = []
        for j in range(len(queries)):
            if topk_value_large[..., j] - topk_value_middle[..., j] > 0:
                pred_mask = predicted_large_objects[topk_indice_large[...,j]][gt_idx]
            else:
                pred_mask = predicted_middle_objects[topk_indice_middle[...,j]][gt_idx]

            intersection = torch.logical_and(pred_mask, gt_masks[j]).sum()
            union = torch.logical_or(pred_mask, gt_masks[j]).sum()
            iou = intersection / union if union > 0 else torch.tensor(0.0)
            ious.append(iou)
            accs.append(1.0 if iou > 0.5 else 0.0)

        print(gt_dir)
        print("queries:", queries)
        print("mean IoU of current frame:", torch.tensor(ious).mean().item())
        print("mean ACC of current frame:", sum(accs) / len(accs))
        all_ious.append(torch.tensor(ious).mean())
        all_accs.append(sum(accs) / len(accs))

    miou = torch.tensor(all_ious).mean().item()
    macc = sum(all_accs) / len(all_accs) if all_accs else 0.0
    print(f"final mIoU: {miou:.4f}")
    print(f"final mACC: {macc:.4f}")
    return miou, macc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True, help="Scene path")
    parser.add_argument("--render_dir", type=str, required=True, help="Results directory path")
    parser.add_argument("--label_dir", type=str, required=True, help="Ground truth label directory path")
    parser.add_argument("--label_format", type=str, default="lerf", choices=["lerf", "3d_ovs"],
                        help="Label format: lerf (frame_XXXXX dirs with .jpg masks) or 3d_ovs (XX dirs with .png masks)")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save results JSON")
    args = parser.parse_args()

    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    large_object_images, middle_object_images, small_object_images = load_images_masks(args.scene)

    print("==================== CLIP Embedding Association ====================")
    torch.cuda.synchronize()
    clip_start = time.time()

    mean_large_object_features, mean_middle_object_features, mean_small_object_features = extract_CLIP_embeddings(
        model, preprocess, large_object_images, middle_object_images, small_object_images
    )

    torch.cuda.synchronize()
    clip_time = time.time() - clip_start
    print(f"CLIP inference time: {clip_time:.1f}s")

    predicted_large_objects, predicted_middle_objects, predicted_small_objects = load_results(args.render_dir)

    print("==================== mIoU / mACC Evaluation ====================")
    gt_data = load_gt_labels(args.label_dir, args.label_format)
    miou, macc = calculate_IoU(gt_data, mean_large_object_features, mean_middle_object_features, mean_small_object_features,
        predicted_large_objects, predicted_middle_objects, predicted_small_objects, model)

    results = {
        "mIoU": miou,
        "mACC": macc,
        "clip_inference_sec": clip_time,
    }

    output_path = args.output_json or os.path.join(args.render_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
