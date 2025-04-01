import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel

import threestudio
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *

from .base_callable import MultiRefProcessor, hash_prompt
from threestudio.utils.misc import get_rank
from tqdm import tqdm

from .utils import hash_prompt, _load_prompt_embedding_v2 as _load_prompt_embedding
from functools import partial


from dataclasses import dataclass, field
from threestudio.models.prompt_processors.base import (
    PromptProcessorOutput,

)
@threestudio.register("dual-stable-diffusion-multi-prompt-processor-callable")
class DualStableDiffusionMultipromptPromptProcessor(MultiRefProcessor):
    @dataclass
    class Config(MultiRefProcessor.Config):
        cache_dir: str = ".threestudio_cache/text_embeddings"  # FIXME: hard-coded path

        use_latent: bool = False
        use_embed_global: bool = True
        use_embed_local: bool = True

        use_local_text_embeddings: bool = True

        pretrained_model_name_or_path: str = "pretrained/stable-diffusion-v1-5"
        pretrained_model_name_or_path_2nd: str = "pretrained/stable-diffusion-2-1-base"

        negative_prompt: str = ""
        negative_prompt_2nd: str = "" # for the second model

    cfg: Config

    ### these functions are unused, kept for debugging ###
    def load_model_text(self) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
        )
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

    def load_text_encoding(
            self,
            prompts: List[str],
            **kwargs
        ) -> None:

        return_dict = {}

        # for the first model
        assert self.cfg.use_latent is False, "Latent embeddings are not supported for text embeddings"
        if self.cfg.use_embed_global:
            text_embedding_global_list = []
        if self.cfg.use_embed_local:
            text_embedding_local_list = []

        for data in map(
                partial(
                    _load_prompt_embedding,
                    cache_dir=self._cache_dir,
                    pretrained_model_name_or_path=self.cfg.pretrained_model_name_or_path,
                    load_embed_global=self.cfg.use_embed_global,
                    load_embed_local=self.cfg.use_embed_local,
                ),
                prompts,
            ):

            if self.cfg.use_embed_global:
                text_embedding_global_list += [data["global_text_embedding"]]
            if self.cfg.use_embed_local:
                text_embedding_local_list += [data["local_text_embedding"]]

        if self.cfg.use_embed_global:
            return_dict["text_embeddings_global"] = text_embedding_global_list
        if self.cfg.use_embed_local:
            return_dict["text_embeddings_local"] = text_embedding_local_list

        # load negative prompt embedding
        if hasattr(self.cfg, "negative_prompt") and self.cfg.negative_prompt is not None:
            data = _load_prompt_embedding(
                self.cfg.negative_prompt,
                cache_dir=self._cache_dir,
                pretrained_model_name_or_path=self.cfg.pretrained_model_name_or_path,
                load_embed_global=self.cfg.use_embed_global,
                load_embed_local=self.cfg.use_embed_local,
            )
            if self.cfg.use_embed_global:
                return_dict["uncond_text_embeddings_global"] = data["global_text_embedding"]
            if self.cfg.use_embed_local:
                return_dict["uncond_text_embeddings_local"] = data["local_text_embedding"]

        # for the second model
        assert self.cfg.use_latent is False, "Latent embeddings are not supported for text embeddings"
        if self.cfg.use_embed_global:
            text_embedding_global_list_2nd = []
        if self.cfg.use_embed_local:
            text_embedding_local_list_2nd = []

        for data in map(
                partial(
                    _load_prompt_embedding,
                    cache_dir=self._cache_dir,
                    pretrained_model_name_or_path=self.cfg.pretrained_model_name_or_path_2nd,
                    load_embed_global=self.cfg.use_embed_global,
                    load_embed_local=self.cfg.use_embed_local,
                ),
                prompts,
            ):

            if self.cfg.use_embed_global:
                text_embedding_global_list_2nd += [data["global_text_embedding"]]
            if self.cfg.use_embed_local:
                text_embedding_local_list_2nd += [data["local_text_embedding"]]

        if self.cfg.use_embed_global:
            return_dict["text_embeddings_global_2nd"] = text_embedding_global_list_2nd
        if self.cfg.use_embed_local:
            return_dict["text_embeddings_local_2nd"] = text_embedding_local_list_2nd

        # load negative prompt embedding
        if hasattr(self.cfg, "negative_prompt_2nd") and self.cfg.negative_prompt_2nd is not None:
            data = _load_prompt_embedding(
                self.cfg.negative_prompt_2nd,
                cache_dir=self._cache_dir,
                pretrained_model_name_or_path=self.cfg.pretrained_model_name_or_path_2nd,
                load_embed_global=self.cfg.use_embed_global,
                load_embed_local=self.cfg.use_embed_local,
            )
            if self.cfg.use_embed_global:
                return_dict["uncond_text_embeddings_global_2nd"] = data["global_text_embedding"]
            if self.cfg.use_embed_local:
                return_dict["uncond_text_embeddings_local_2nd"] = data["local_text_embedding"]

        # load the default unconditional text embeddings
        data = _load_prompt_embedding(
            "", # empty prompt
            cache_dir=self._cache_dir,
            pretrained_model_name_or_path=self.cfg.pretrained_model_name_or_path_2nd,
            load_embed_global=self.cfg.use_embed_global,
            load_embed_local=self.cfg.use_embed_local,
        )
        if self.cfg.use_embed_global:
            return_dict["default_uncond_text_embeddings_global"] = data["global_text_embedding"]
        if self.cfg.use_embed_local:
            return_dict["default_uncond_text_embeddings_local"] = data["local_text_embedding"]

        return return_dict
    
    def __call__(
        self,
        image_paths: Optional[Union[str, List[str]]] = None,
        prompts: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> PromptProcessorOutput:
        
        if image_paths is not None:
            raise NotImplementedError("Image processing is not supported for this processor")
        
        if prompts is not None:
            assert image_paths is None, "Cannot process both image and text at the same time"
            if isinstance(prompts, str):
                prompts = [prompts]

            prompt_args = self.load_text_encoding(prompts, **kwargs)

            # update other arguments
            if hasattr(self.cfg, "use_local_text_embeddings"):
                prompt_args["use_local_text_embeddings"] = self.cfg.use_local_text_embeddings

            # # deprecated
            # return MultiRefProcessorOutput4Text_DualSD(
            #     device=self.device,
            #     **prompt_args,
            # )
            
            return MultiRefProcessorOutput4Text_DualSD_v2(
                device=self.device,
                **prompt_args,
            )
        
@dataclass
class MultiRefProcessorOutput4Text_DualSD:
    text_embeddings_global: Optional[List[Float[Tensor, "B ..."]]] = None
    text_embeddings_local: Optional[List[Float[Tensor, "B ..."]]] = None
    uncond_text_embeddings_global: Optional[Float[Tensor, "B ..."]] = None
    uncond_text_embeddings_local: Optional[Float[Tensor, "B ..."]] = None

    text_embeddings_global_2nd: Optional[List[Float[Tensor, "B ..."]]] = None
    text_embeddings_local_2nd: Optional[List[Float[Tensor, "B ..."]]] = None
    uncond_text_embeddings_global_2nd: Optional[Float[Tensor, "B ..."]] = None
    uncond_text_embeddings_local_2nd: Optional[Float[Tensor, "B ..."]] = None

    # must have the following attributes
    use_local_text_embeddings: bool = False
    device: str = "cuda"

    appendable_attributes: list = field(
        default_factory=lambda: [
            "text_embeddings_global",
            "text_embeddings_local",
            "text_embeddings_global_2nd",
            "text_embeddings_local_2nd",
        ]
    )

    def get_uncond_text_embeddings(self):
        raise NotImplementedError("Unconditional text embeddings are not supported for this processor")
        
    def get_text_embeddings(
        self,
        **kwargs
    ):
        if "view_dependent_prompting" in kwargs and kwargs["view_dependent_prompting"]:
            raise NotImplementedError("View-dependent prompting is not supported for text embeddings")

        batch_size = len(self.text_embeddings_global)

        # for the first model
        text_embeddings = torch.stack(
            self.text_embeddings_local,
            dim=0
        ).to(self.device)

        uncond_text_embeddings_local = self.uncond_text_embeddings_local
        if isinstance(self.uncond_text_embeddings_local, List):
            uncond_text_embeddings_local = self.uncond_text_embeddings_local[0]
        uncond_text_embeddings_local = uncond_text_embeddings_local[None, :, :].repeat(
                batch_size, 1, 1
            ).to(self.device)
        
        # for the second model
        text_embeddings_2nd = torch.stack(
            self.text_embeddings_local_2nd,
            dim=0
        ).to(self.device)

        uncond_text_embeddings_local_2nd = self.uncond_text_embeddings_local_2nd
        if isinstance(self.uncond_text_embeddings_local_2nd, List):
            uncond_text_embeddings_local_2nd = self.uncond_text_embeddings_local_2nd[0]
        uncond_text_embeddings_local_2nd = uncond_text_embeddings_local_2nd[None, :, :].repeat(
                batch_size, 1, 1
            ).to(self.device)
        
        return torch.cat(
            [
                text_embeddings,
                uncond_text_embeddings_local,
            ],
            dim=0,
        ), torch.cat(
            [
                text_embeddings_2nd,
                uncond_text_embeddings_local_2nd,
            ],
            dim=0,
        )
        

    def get_global_text_embeddings(
        self,
        use_local_text_embeddings: Optional[bool] = None,
    ):
        raise NotImplementedError("Global text embeddings are not supported for this processor")


@dataclass
class MultiRefProcessorOutput4Text_DualSD_v2(MultiRefProcessorOutput4Text_DualSD):

    # sort of a hack to make the processor compatible with the unconditional prompt of ""
    default_uncond_text_embeddings_global: Optional[Float[Tensor, "B ..."]] = None
    default_uncond_text_embeddings_local: Optional[Float[Tensor, "B ..."]] = None


    def get_text_embeddings(
        self,
        use_default_neg = False,
        **kwargs
    ):
        if "view_dependent_prompting" in kwargs and kwargs["view_dependent_prompting"]:
            raise NotImplementedError("View-dependent prompting is not supported for text embeddings")

        batch_size = len(self.text_embeddings_global)

        # for the first model
        text_embeddings = torch.stack(
            self.text_embeddings_local,
            dim=0
        ).to(self.device)

        uncond_text_embeddings_local = self.uncond_text_embeddings_local
        if isinstance(self.uncond_text_embeddings_local, List):
            uncond_text_embeddings_local = self.uncond_text_embeddings_local[0]
        uncond_text_embeddings_local = uncond_text_embeddings_local[None, :, :].repeat(
                batch_size, 1, 1
            ).to(self.device)
        
        # for the second model
        text_embeddings_2nd = torch.stack(
            self.text_embeddings_local_2nd,
            dim=0
        ).to(self.device)

        # can use the default unconditional text embeddings
        if use_default_neg:
            uncond_text_embeddings_local_2nd = self.default_uncond_text_embeddings_local
            if isinstance(self.default_uncond_text_embeddings_local, List):
                uncond_text_embeddings_local_2nd = self.default_uncond_text_embeddings_local[0]
            uncond_text_embeddings_local_2nd = uncond_text_embeddings_local_2nd[None, :, :].repeat(
                    batch_size, 1, 1
                ).to(self.device)
        else:
            uncond_text_embeddings_local_2nd = self.uncond_text_embeddings_local_2nd
            if isinstance(self.uncond_text_embeddings_local_2nd, List):
                uncond_text_embeddings_local_2nd = self.uncond_text_embeddings_local_2nd[0]
            uncond_text_embeddings_local_2nd = uncond_text_embeddings_local_2nd[None, :, :].repeat(
                    batch_size, 1, 1
                ).to(self.device)

        
        return torch.cat(
            [
                text_embeddings,
                uncond_text_embeddings_local,
            ],
            dim=0,
        ), torch.cat(
            [
                text_embeddings_2nd,
                uncond_text_embeddings_local_2nd,
            ],
            dim=0,
        )
