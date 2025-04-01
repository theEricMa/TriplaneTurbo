"""
Code for CLIPScore (https://arxiv.org/abs/2104.08718)
@inproceedings{hessel2021clipscore,
  title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
  author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
  booktitle={EMNLP},
  year={2021}
}
"""
import argparse
import collections
import json
import os
import pathlib

# import generation_eval_utils
import pprint
import warnings

import numpy as np
import sklearn.preprocessing

# import clip  # Remove this import
import torch
import tqdm
from packaging import version
from PIL import Image
from sklearn.preprocessing import normalize
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

# Add transformers imports
from transformers import CLIPModel, CLIPProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir",
        type=str,
        help="Candidates json mapping from image_id --> candidate.",
    )

    args = parser.parse_args()

    return args


class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix="A photo depicts"):
        self.data = data
        self.prefix = prefix
        # Initialize the processor here
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __getitem__(self, idx):
        c_data = self.data[idx]
        # Replace clip.tokenize with the HuggingFace processor
        inputs = self.processor(
            text=self.prefix + c_data,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {"caption": inputs.input_ids.squeeze()}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # Use HuggingFace processor instead of custom transforms
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        # Use processor for image preprocessing
        inputs = self.processor(images=image, return_tensors="pt")
        return {"image": inputs.pixel_values.squeeze()}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=256, num_workers=8):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    all_text_features = []

    # 每次处理一批文本，确保它们被正确填充
    for i in tqdm.tqdm(range(0, len(captions), batch_size)):
        batch_captions = captions[i : i + batch_size]
        inputs = processor(
            text=batch_captions, return_tensors="pt", padding=True, truncation=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            all_text_features.append(text_features.cpu().numpy())

    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b["image"].to(device)
            image_features = model.get_image_features(b)
            all_image_features.append(image_features.cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def get_clip_score(model, images, candidates, device, w=2.5):
    """
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    """
    if isinstance(images, list):
        # need to extract image features
        images = extract_all_images(images, model, device)

    candidates = extract_all_captions(candidates, model, device)

    # as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse("1.21"):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        warnings.warn(
            "due to a numerical instability, new numpy normalization is slightly different than paper results. "
            "to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3."
        )
        images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
        candidates = candidates / np.sqrt(
            np.sum(candidates**2, axis=1, keepdims=True)
        )

    per = w * np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates


def get_clip_recall(model, images, all_candidates, correct_idxs, device, w=2.5):
    if isinstance(images, list):
        # need to extract image features
        images = extract_all_images(images, model, device)

    all_candidates = extract_all_captions(all_candidates, model, device)

    # as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse("1.21"):
        images = sklearn.preprocessing.normalize(images, axis=1)
        all_candidates = sklearn.preprocessing.normalize(all_candidates, axis=1)
    else:
        warnings.warn(
            "due to a numerical instability, new numpy normalization is slightly different than paper results. "
            "to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3."
        )
        images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
        all_candidates = all_candidates / np.sqrt(
            np.sum(all_candidates**2, axis=1, keepdims=True)
        )

    mat = images @ all_candidates.T
    mat = np.clip(mat, 0, None).argmax(axis=-1)
    recall = (mat == correct_idxs).mean()

    return recall


def is_image(image):
    return image.endswith(".jpg") or image.endswith(".png")


def main():
    args = parse_args()

    prompts = os.listdir(args.result_dir)
    prompts = [
        prompt
        for prompt in prompts
        if os.path.isdir(os.path.join(args.result_dir, prompt))
    ]

    image_paths = []
    candidates = []
    for prompt in prompts:
        sub_dir = os.path.join(args.result_dir, prompt)
        for image in os.listdir(sub_dir):
            if is_image(image):
                image_paths.append(os.path.join(sub_dir, image))
                candidates.append(prompt.replace("_", " "))

    image_ids = [pathlib.Path(path).stem for path in image_paths]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn(
            "CLIP runs in full float32 on CPU. Results in paper were computed on GPU, which uses float16. "
            "If you're reporting results on CPU, please note this when you report."
        )

    # Replace CLIP model loading with HuggingFace model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()

    image_feats = extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=8
    )

    # get image-text clipscore
    _, per_instance_image_text, candidate_feats = get_clip_score(
        model, image_feats, candidates, device
    )

    scores = {
        image_id: {"CLIPScore": float(clipscore)}
        for image_id, clipscore in zip(image_ids, per_instance_image_text)
    }
    print(
        "CLIPScore: {:.4f}".format(np.mean([s["CLIPScore"] for s in scores.values()]))
    )

    path = os.path.join(args.result_dir, "clip-scores.txt")
    with open(path, "w") as f:
        f.write(json.dumps(scores))
    print(f"Saved scores to {path}")

    all_candidates = list(set(candidates))
    # find the correct index for each candidate
    candidate2idx = {c: i for i, c in enumerate(all_candidates)}
    correct_idxs = [candidate2idx[c] for c in candidates]

    recall = get_clip_recall(model, image_feats, all_candidates, correct_idxs, device)
    print(f"Recall@1: {recall}")

    # save the recall
    path = os.path.join(args.result_dir, "clip-recall.txt")
    with open(path, "w") as f:
        f.write(f"Recall@1: {recall}")
    print(f"Saved recall to {path}")


if __name__ == "__main__":
    main()
