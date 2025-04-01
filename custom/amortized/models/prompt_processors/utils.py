import torch   
import os
import hashlib

def hash_prompt(model: str, prompt: str, type: str) -> str:
    identifier = f"{model}-{prompt}-{type}"
    return hashlib.md5(identifier.encode()).hexdigest()

def hash_image(model: str, image_path: str, size: tuple, type: str) -> str:
    size = f"{size[0]}x{size[1]}"
    identifier = f"{model}-{image_path}-{size}-{type}"
    return hashlib.md5(identifier.encode()).hexdigest()

def load_from_cache_text(cache_dir, pretrained_model_name_or_path, prompt, load_local=False, load_global=False, ):
        # load global text embedding
        if load_global:
            cache_path_global = os.path.join(
                cache_dir,
                f"{hash_prompt(pretrained_model_name_or_path, prompt, 'global')}.pt",
            )
            if not os.path.exists(cache_path_global):
                raise FileNotFoundError(
                    f"Global Text embedding file {cache_path_global} for model {pretrained_model_name_or_path} and prompt [{prompt}] not found."
                )
            global_text_embedding = torch.load(cache_path_global, map_location='cpu')

        # load local text embedding
        if load_local:
            cache_path_local = os.path.join(
                cache_dir,
                f"{hash_prompt(pretrained_model_name_or_path, prompt, 'local')}.pt",
            )
            if not os.path.exists(cache_path_local):
                raise FileNotFoundError(
                    f"Local Text embedding file {cache_path_local} for model {pretrained_model_name_or_path} and prompt [{prompt}] not found."
                )
            local_text_embedding = torch.load(cache_path_local, map_location='cpu')

        # the return value depends on the flags
        if load_local and load_global:
            return global_text_embedding, local_text_embedding
        elif load_local:
            return local_text_embedding
        elif load_global:
            return global_text_embedding

def _load_prompt_embedding(args):
    """
        Load the global/local text embeddings for a single prompt
        from cache into memory
    """
    prompt, prompt_vds, cache_dir, pretrained_model_name_or_path = args
    global_text_embeddings, local_text_embeddings = load_from_cache_text(
        cache_dir, pretrained_model_name_or_path, prompt, 
        load_global=True, load_local=True
    )
    text_embeddings_vd = torch.stack(
        [load_from_cache_text(
            cache_dir, pretrained_model_name_or_path, prompt, 
            load_global=False, load_local=True) for prompt in prompt_vds], dim=0 # we don't need local text embeddings for view-dipendent conditional generation
    )
    return prompt, global_text_embeddings, local_text_embeddings, text_embeddings_vd

def _load_prompt_embedding_v2(
        prompt, cache_dir, pretrained_model_name_or_path,
        load_embed_global=True, load_embed_local=True,
):
    """
        Load the global/local text embeddings for a single prompt
        from cache into memory
    """
    return_dict = {
        "prompt": prompt,
    }
    if load_embed_global:
        cache_path = os.path.join(
            cache_dir,
            f"{hash_prompt(pretrained_model_name_or_path, prompt, 'global')}.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Global Text embedding file {cache_path} for model {pretrained_model_name_or_path} and prompt [{prompt}] not found."
            )
        global_text_embedding = torch.load(cache_path, map_location='cpu')
        return_dict["global_text_embedding"] = global_text_embedding

    if load_embed_local:
        cache_path = os.path.join(
            cache_dir,
            f"{hash_prompt(pretrained_model_name_or_path, prompt, 'local')}.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Local Text embedding file {cache_path} for model {pretrained_model_name_or_path} and prompt [{prompt}] not found."
            )
        local_text_embedding = torch.load(cache_path, map_location='cpu')
        return_dict["local_text_embedding"] = local_text_embedding

    return return_dict


def _load_image_embedding(
        image_path, cache_dir, pretrained_model_name_or_path, 
        load_latent=True, load_embed_global=True, load_embed_local=False,
        image_size = 256
    ):
    """
        Load the image encodings for a single image
        from cache into memory
    """

    return_dict = {
        "image_path": image_path,
    }
    if load_latent:
        cache_path = os.path.join(
            cache_dir,
            f"{hash_image(pretrained_model_name_or_path, image_path, image_size, 'latent')}.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Image latent file {cache_path} for model {pretrained_model_name_or_path} and image [{image_path}] with size [{image_size}] not found."
            )
        image_latent = torch.load(cache_path, map_location='cpu')
        return_dict["image_latent"] = image_latent


    if load_embed_global:
        cache_path = os.path.join(
            cache_dir,
            f"{hash_image(pretrained_model_name_or_path, image_path, image_size, 'global')}.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Image embedding file {cache_path} for model {pretrained_model_name_or_path} and image [{image_path}] with size [{image_size}] not found."
            )
        image_embedding_global = torch.load(cache_path, map_location='cpu')
        return_dict["image_embedding_global"] = image_embedding_global

    if load_embed_local:
        cache_path = os.path.join(
            cache_dir,
            f"{hash_image(pretrained_model_name_or_path, image_path, image_size, 'local')}.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Image embedding file {cache_path} for model {pretrained_model_name_or_path} and image [{image_path}] with size [{image_size}] not found."
            )
        image_embedding_local = torch.load(cache_path, map_location='cpu')
        return_dict["image_embedding_local"] = image_embedding_local

    return return_dict