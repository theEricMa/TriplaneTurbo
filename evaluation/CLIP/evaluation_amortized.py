import argparse
import os
import ssl

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

ssl._create_default_https_context = ssl._create_unverified_context


def is_image(name):
    name = name.lower()
    return (
        name.endswith(".png")
        or name.endswith(".jpg")
        or name.endswith(".jpeg")
        or name.endswith(".bmp")
    )


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [
            torchvision.transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            )
        ]
    return torchvision.transforms.Compose(transform_list)


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        size = image.size
        image = image.crop((0, 0, size[1], size[1]))
        image = image.resize((224, 224), resample=Image.BILINEAR)
        if self.transform is not None:
            image = self.transform(image)
        return image


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--result_dir", type=str, default=None)
    argparser.add_argument("--num_images", type=int, default=100)
    argparser.add_argument("--device", type=str, default="cuda:0")
    argparser.add_argument("--batch_size", type=int, default=120)
    args = argparser.parse_args()

    device = args.device
    batch_size = args.batch_size
    # list all prompts the result_dir
    prompts = os.listdir(args.result_dir)
    # filter out the files, leaving only the directories
    prompts = [
        prompt
        for prompt in prompts
        if os.path.isdir(os.path.join(args.result_dir, prompt))
    ]

    # remove the '_' in the prompts with ' '
    prompts_inputs = [prompt.replace("_", " ") for prompt in prompts]

    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-base-patch32", force_download=False
    )
    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch32", force_download=False
    ).to(device)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch32", force_download=False
    ).to(device)

    text_inputs = tokenizer(
        prompts_inputs,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_inputs = text_inputs.input_ids.to(device)

    text_features = []
    for i in range(0, len(text_inputs), batch_size):
        text_batch = text_inputs[i : i + batch_size]
        with torch.no_grad():
            text_feature = text_encoder(text_batch)[0]
            text_features.append(text_feature)
    text_features_tensor = torch.cat(text_features, dim=0)
    text_features_tensor = text_features_tensor / text_features_tensor.norm(
        p=2, dim=-1, keepdim=True
    )

    # record the results
    similarity_dict = {}
    recall_dict = {}

    for idx, (prompt, prompt_input) in enumerate(
        zip(prompts, tqdm(prompts_inputs, desc="Looping over prompts"))
    ):
        sub_dir = os.path.join(args.result_dir, prompt)
        images = os.listdir(sub_dir)
        images = [os.path.join(sub_dir, image) for image in images if is_image(image)]
        dataset = ImageDataset(images, transform=get_tensor_clip())
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=32
        )

        for image_batch in tqdm(dataloader, desc="Looping over images"):
            # get the image features
            with torch.no_grad():
                image_batch = image_batch.to(device)
                image_features = image_encoder(image_batch)[0]  # (batch_size, 768)

                #  fetch the text features
                text_features = text_features_tensor[idx].expand_as(
                    image_features
                )  # (batch_size, 768)

                # compute the similarity
                image_features = image_features / image_features.norm(
                    p=2, dim=-1, keepdim=True
                )
                similarity = image_features @ text_features.T
                # only keep the diagonal elements
                similarity = similarity.diag()

                # compute the recall@1
                text_probs = (100.0 * image_features @ text_features_tensor.T).softmax(
                    dim=-1
                )
                is_correct = (text_probs.argmax(dim=-1) == idx).cpu().numpy().tolist()

            if prompt not in similarity_dict:
                similarity_dict[prompt] = []
                recall_dict[prompt] = []

            similarity_dict[prompt].extend(similarity.detach().cpu().numpy().tolist())
            recall_dict[prompt].extend(is_correct)

        if prompt not in similarity_dict:
            continue

        # compute the average similarity and recall@1
        similarity_dict[prompt] = sum(similarity_dict[prompt]) / len(
            similarity_dict[prompt]
        )
        recall_dict[prompt] = sum(recall_dict[prompt]) / len(recall_dict[prompt])

    # compute the average similarity and recall@1
    similarity = sum(similarity_dict.values()) / len(similarity_dict)
    recall = sum(recall_dict.values()) / len(recall_dict)

    # record the results
    path = os.path.join(args.result_dir, "similarity.txt")
    with open(path, "w") as f:
        for prompt in similarity_dict:
            f.write(f"{prompt}: {similarity_dict[prompt]}\n")
        f.write(f"avgerage: {similarity}\n")

    # record the results
    path = os.path.join(args.result_dir, "recall.txt")
    with open(path, "w") as f:
        for prompt in recall_dict:
            f.write(f"{prompt}: {recall_dict[prompt]}\n")
        f.write(f"avgerage: {recall}\n")

    print(f"Average similarity: {similarity}")
    print(f"Average recall@1: {recall}")

    print("Results saved to: ", args.result_dir)
