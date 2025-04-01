import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel

import threestudio
from .base_dual_uncond import MultiPromptProcessor, hash_prompt
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *

@threestudio.register("stable-diffusion-multi-prompt-processor-dual-uncond")
class StableDiffusionMultipromptPromptProcessor(MultiPromptProcessor):
    @dataclass
    class Config(MultiPromptProcessor.Config):
        pass

    cfg: Config

    ### these functions are unused, kept for debugging ###
    def configure_text_encoder(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder"
        ).to(self.device)

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.text_encoder
        cleanup()

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir, tokenizer = None, text_encoder = None):
        if tokenizer is None:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, subfolder="tokenizer"
            )

        if text_encoder is None:
            text_encoder = CLIPTextModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="text_encoder",
                device_map="auto",
            )
        
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

        del text_encoder
        cleanup()