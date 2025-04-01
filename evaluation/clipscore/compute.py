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

import clip
import numpy as np
import sklearn.preprocessing
import torch
import tqdm
from packaging import version
from PIL import Image
from sklearn.preprocessing import normalize
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor


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
        # if self.prefix[-1] != ' ':
        #     self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {"caption": c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose(
            [
                Resize(n_px, interpolation=Image.BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {"image": image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=256, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    all_text_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b["caption"].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
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
            if device == "cuda":
                b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
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
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
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
