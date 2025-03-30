import json
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel

import threestudio
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *

from .base_callable import MultiRefProcessor, hash_prompt
from threestudio.utils.misc import get_rank
from tqdm import tqdm

@threestudio.register("stable-diffusion-multi-prompt-processor-callable")
class StableDiffusionMultipromptPromptProcessor(MultiRefProcessor):
    @dataclass
    class Config(MultiRefProcessor.Config):
        cache_dir: str = ".threestudio_cache/text_embeddings"  # FIXME: hard-coded path

        use_latent: bool = False
        use_embed_global: bool = True
        use_embed_local: bool = True
        use_local_text_embeddings: bool = True

        negative_prompt: str = ""
        # FIXME: hard-coded list
        other_prompts: list = field(
            default_factory=lambda: [
                "ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"
        ])

    cfg: Config

    ### these functions are unused, kept for debugging ###
    def load_model_text(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder",
        ).to(self.device)

        for p in text_encoder.parameters():
            p.requires_grad_(False)

        return {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
        }


    def func_text(
        self,
        pretrained_model_name_or_path: str,
        prompts,
        cache_dir: str,
        tokenizer: Optional[AutoTokenizer] = None,
        text_encoder: Optional[CLIPTextModel] = None,
    ) -> Any:
        
        if tokenizer is None or text_encoder is None:
            modules = self.load_model_text()
            tokenizer = modules.pop("tokenizer")
            text_encoder = modules.pop("text_encoder")

        
        if type(prompts) == str:
            prompts = [prompts]

        with torch.no_grad():
            tokens = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            # avoid exceeding max_length
            tokens.input_ids = tokens.input_ids[:, :tokenizer.model_max_length]
            
            outputs = text_encoder(tokens.input_ids.to(text_encoder.device))
            # we need both the local and global text embeddings
            locals_text_embeddings, globla_text_embeddings= outputs[0], outputs[1]

        for prompt, globla_text_embedding, locals_text_embedding in zip(prompts, globla_text_embeddings, locals_text_embeddings):
            # save the global text embeddings
            torch.save(
                globla_text_embedding.cpu(), # [0] is to remove the batch dimension
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt, 'global')}.pt",
                ),
            )

            # save the local text embeddings
            torch.save(
                locals_text_embedding.cpu(), # [0] is to remove the batch dimension
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt, 'local')}.pt",
                ),
            )


    def spawn_func_text(
        self, 
        args,
    ):
        pretrained_model_name_or_path, prompt_list, cache_dir = args
        modules = self.load_model_text()
        tokenizer = modules.pop("tokenizer")
        text_encoder = modules.pop("text_encoder")

        batch_size = 32 # hard coded batch size
        rank = get_rank()
        for i in tqdm(
            range(0, len(prompt_list), batch_size),
            desc="Saving text embeddings in GPU {}".format(rank),
        ):

            prompts = prompt_list[i:i+batch_size]
            with torch.no_grad():
                input_ids = []
                for prompt in prompts:
                    tokens = tokenizer(
                        [prompt],
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt",
                    )
                    # avoid exceeding max_length
                    tokens.input_ids = tokens.input_ids[:, :tokenizer.model_max_length]
                    input_ids.append(tokens.input_ids)
                
                outputs = text_encoder(torch.cat(input_ids, dim=0).to(text_encoder.device))
                # we need both the local and global text embeddings
                locals_text_embeddings, globla_text_embeddings= outputs[0], outputs[1]

            for prompt, globla_text_embedding, locals_text_embedding in zip(prompts, globla_text_embeddings, locals_text_embeddings):
                # save the global text embeddings
                torch.save(
                    globla_text_embedding.cpu(), # [0] is to remove the batch dimension
                    os.path.join(
                        cache_dir,
                        f"{hash_prompt(pretrained_model_name_or_path, prompt, 'global')}.pt",
                    ),
                )

                # save the local text embeddings
                torch.save(
                    locals_text_embedding.cpu(), # [0] is to remove the batch dimension
                    os.path.join(
                        cache_dir,
                        f"{hash_prompt(pretrained_model_name_or_path, prompt, 'local')}.pt",
                    ),
                )
                
        del text_encoder
        del tokenizer
        cleanup()