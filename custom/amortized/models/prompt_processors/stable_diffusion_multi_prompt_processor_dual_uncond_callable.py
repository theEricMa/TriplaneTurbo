import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel

import threestudio
from .base_dual_uncond_callable import MultiPromptProcessor, hash_prompt
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *

from threestudio.utils.misc import get_rank
from tqdm import tqdm
@threestudio.register("stable-diffusion-multi-prompt-processor-dual-uncond-callable")
class StableDiffusionMultipromptCallableProcessor(MultiPromptProcessor):
    @dataclass
    class Config(MultiPromptProcessor.Config):
        pass

    cfg: Config

    @staticmethod
    def func(pretrained_model_name_or_path, prompts, cache_dir, tokenizer = None, text_encoder = None):
        
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

        iterator = zip(prompts, globla_text_embeddings, locals_text_embeddings)
        for prompt, globla_text_embedding, locals_text_embedding in iterator:
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

    @staticmethod
    def spawn_func(args):
        pretrained_model_name_or_path, prompt_list, cache_dir = args

        # load the tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )

        # load the text encoder
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            device_map="auto",
        )
        
        if type(prompt_list) == str:
            prompt_list = [prompt_list]

        batch_size = 32 # hard coded batch size
        rank = get_rank(opposite=True)
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