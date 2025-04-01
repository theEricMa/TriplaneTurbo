import json
import os
from dataclasses import dataclass
from dataclasses import field

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import *
from threestudio.utils.misc import barrier, cleanup

from functools import partial

from threestudio.models.prompt_processors.base import (
    DirectionConfig, shift_azimuth_deg, PromptProcessorOutput,
    shifted_expotional_decay
)
from threestudio.utils.misc import get_rank

from itertools import cycle
from .utils import hash_prompt, hash_image, _load_image_embedding, _load_prompt_embedding_v2 as _load_prompt_embedding

from functools import partial
from tqdm import tqdm


##############################################

class MultiRefProcessor(BaseObject):
    @dataclass
    class Config(BaseObject.Config):

        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        cache_dir: str = ".threestudio_cache/image_encodings"  # FIXME: hard-coded path

        use_cache: bool = True
        spawn: bool = False

        # the following attributes are for image processing
        image_root_dir: str = ""

        use_latent: bool = True
        use_embed_global: bool = True
        use_embed_local: bool = False

        default_prompt: str = ""

    cfg: Config

    def __reduce__(self) -> str | tuple[Any, ...]:
        return (self.__class__, (self.configure,))
    
    def configure(self) -> None:
        self._cache_dir = self.cfg.cache_dir

    @staticmethod
    def func_image(pretrained_model_name_or_path, image_paths, cache_dir, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def spawn_func_image(pretrained_model_name_or_path, image_paths, cache_dir, **kwargs):
        raise NotImplementedError

    @staticmethod
    def func_text(pretrained_model_name_or_path, prompts, cache_dir, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def spawn_func_text(pretrained_model_name_or_path, prompts, cache_dir, **kwargs):
        raise NotImplementedError
    
    def load_model_image(
        self,
    ):
        """
        Load model for image processing
        """
        pass

    def load_model_text(
        self,
    ):
        """
        Load model for text processing
        """
        pass

    def load_image(
        self,
    ):
        """
        Preprocess image before passing to the model
        """
        pass

    def prepare_text_embeddings(
        self,
        all_prompts: List[str],
        **kwargs
    ) -> None:

        os.makedirs(self._cache_dir, exist_ok=True)

        rank = get_rank(opposite=True)
        # find all available gpus via world_size
        num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", None).split(","))
        if num_gpus > 1:
            # each process only has a subset of the prompt library!
            all_prompts = all_prompts[rank::num_gpus]

        # add negative prompt to the list for rank 0
        if rank == 0:
            if hasattr(self.cfg, "default_prompt") and self.cfg.default_prompt is not None:
                all_prompts += [self.cfg.default_prompt]
            if hasattr(self.cfg, "negative_prompt") and self.cfg.negative_prompt is not None:
                all_prompts += [self.cfg.negative_prompt]
            # add other prompts, if any
            if hasattr(self.cfg, "other_prompts") and self.cfg.other_prompts is not None:
                try:
                    all_prompts += self.cfg.other_prompts
                except:
                    all_prompts += [self.cfg.other_prompts]

        prompts_to_process = []
        for prompt in tqdm(all_prompts, desc="Checking existing text embeddings"):
            if self.cfg.use_cache:

                # some text encodings might be already in cache
                embed_local_pass = True
                if self.cfg.use_embed_local:
                    cache_path = os.path.join(
                        self._cache_dir,
                        f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt, 'local')}.pt",
                    )
                    if not os.path.exists(cache_path):
                        embed_local_pass = False

                embed_global_pass = True
                if self.cfg.use_embed_global:
                    cache_path = os.path.join(
                        self._cache_dir,
                        f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt, 'global')}.pt",
                    )
                    if not os.path.exists(cache_path):
                        embed_global_pass = False

                if embed_local_pass and embed_global_pass:
                    threestudio.debug(
                        f"Text embeddings for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] are already in cache, skip processing."
                    )
                    continue

            prompts_to_process.append(prompt)

        if len(prompts_to_process) > 0:

            if self.cfg.spawn:
                self.spawn_func_text(
                    (
                        self.cfg.pretrained_model_name_or_path,
                        prompts_to_process,
                        self._cache_dir,
                    )
                )

            else:
                # load tokenizer and text_encoder in the main process
                # tokenizer, text_encoder = self.load_model_text()
                modules = self.load_model_text()

                # single process
                for prompt in tqdm(prompts_to_process, desc="Processing prompts"):
                    self.func_text(
                        self.cfg.pretrained_model_name_or_path,
                        prompt,
                        self._cache_dir,
                        **modules,
                    )

                # no need to keep tokenizer and text_encoder in memory
                for module in modules:
                    del module
                cleanup()

    #@rank_zero_only, deprecated when each process has its own cache
    def prepare_image_encodings(
            self, 
            all_image_paths: List[str],
            **kwargs
        ) -> None:

        os.makedirs(self._cache_dir, exist_ok=True)

        rank = get_rank(opposite=True)
        num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", None).split(","))
        if num_gpus > 1:
            # each process only has a subset of the prompt library!
            all_image_paths = all_image_paths[rank::num_gpus]

        image_paths_to_process = []
        for image_path in tqdm(all_image_paths, desc="Checking existing image embeddings"):
            
            assert os.path.exists(
                os.path.join(self.cfg.image_root_dir, image_path)
            ), f"Image path {image_path} does not exist"

            if self.cfg.use_cache:
                
                # some image encodings are already in cache
                # do not process them

                latent_pass = True
                if self.cfg.use_latent:
                    cache_path_latent = os.path.join(
                        self._cache_dir,
                        f"{hash_image(self.cfg.pretrained_model_name_or_path, image_path, self.cfg.image_size,'latent')}.pt",
                    )
                    if not os.path.exists(cache_path_latent):
                        latent_pass = False

                embed_global_pass = True
                if self.cfg.use_embed_global:
                    cache_path_embed = os.path.join(
                        self._cache_dir,
                        f"{hash_image(self.cfg.pretrained_model_name_or_path, image_path, self.cfg.image_size,'global')}.pt",
                    )
                    if not os.path.exists(cache_path_embed):
                        embed_global_pass = False

                embed_local_pass = True
                if self.cfg.use_embed_local:
                    cache_path_embed = os.path.join(
                        self._cache_dir,
                        f"{hash_image(self.cfg.pretrained_model_name_or_path, image_path, self.cfg.image_size,'local')}.pt",
                    )
                    if not os.path.exists(cache_path_embed):
                        embed_local_pass = False

                if latent_pass and embed_global_pass and embed_local_pass:
                    threestudio.debug(
                        f"Image embeddings for model {self.cfg.pretrained_model_name_or_path} and image [{image_path}] are already in cache, skip processing."
                    )
                    continue
                
            image_paths_to_process.append(image_path)

        if len(image_paths_to_process) > 0:

            if self.cfg.spawn:
                self.spawn_func_image(
                    (
                        self.cfg.pretrained_model_name_or_path,
                        image_paths_to_process,
                        self._cache_dir,
                    )
                )

            else:
                # load latent and embed encoder in the main process
                # model, latent_encoder, embedding_encoder = self.load_model_image()
                modules = self.load_model_image()

                # single process
                for image_path in tqdm(image_paths_to_process, desc="Processing refs"):
                    self.func_image(
                        self.cfg.pretrained_model_name_or_path,
                        image_path,
                        self._cache_dir,
                        **modules,
                    )

                # no need to keep tokenizer and text_encoder in memory
                for module in modules:
                    del module
                cleanup()

    def load_text_encoding(
            self,
            prompts: List[str],
            **kwargs
        ) -> None:

        if self.cfg.use_embed_global:
            text_embedding_global_list = []
        if self.cfg.use_embed_local:
            text_embedding_local_list = []

        # for debugging and single-gpu, single process
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

        return_dict = {}
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

        return return_dict

    def load_image_encoding(
            self, 
            image_path_batch: List[str],
            **kwargs
        ) -> None:

        if self.cfg.use_latent:
            image_latent_list = []
        if self.cfg.use_embed_global:
            image_embedding_global_list = []
        if self.cfg.use_embed_local:
            image_embedding_local_list = []

        # for debugging and single-gpu, single process
        for data in map(
                partial(
                    _load_image_embedding,
                    cache_dir=self._cache_dir,
                    pretrained_model_name_or_path=self.cfg.pretrained_model_name_or_path,
                    load_latent=self.cfg.use_latent,
                    load_embed_global=self.cfg.use_embed_global,
                    load_embed_local=self.cfg.use_embed_local,
                    image_size = self.cfg.image_size if hasattr(self.cfg, "image_size") else 256,
                ),
                image_path_batch,
            ):

            if self.cfg.use_latent:
                image_latent_list += [data["image_latent"]]
            if self.cfg.use_embed_global:
                image_embedding_global_list += [data["image_embedding_global"]]
            if self.cfg.use_embed_local:
                image_embedding_local_list += [data["image_embedding_local"]]

        return_dict = {}
        if self.cfg.use_latent:
            return_dict["image_latents"] = image_latent_list
        if self.cfg.use_embed_global:
            return_dict["image_embeddings_global"] = image_embedding_global_list
        if self.cfg.use_embed_local:
            return_dict["image_embeddings_local"] = image_embedding_local_list
        return return_dict

    def __call__(
        self,
        image_paths: Optional[Union[str, List[str]]] = None,
        prompts: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> PromptProcessorOutput:
        
        if image_paths is not None:
            assert prompts is None, "Cannot process both image and text at the same time"
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            
            image_args  = self.load_image_encoding(image_paths, **kwargs)
            prompt_args = self.load_text_encoding(
                [self.cfg.default_prompt],
                **kwargs
            )
            return MultiRefProcessorOutput4Image(
                device=self.device,
                **image_args,
                **prompt_args,
            )
        
        if prompts is not None:
            assert image_paths is None, "Cannot process both image and text at the same time"
            if isinstance(prompts, str):
                prompts = [prompts]

            prompt_args = self.load_text_encoding(prompts, **kwargs)

            # update other arguments
            if hasattr(self.cfg, "use_local_text_embeddings"):
                prompt_args["use_local_text_embeddings"] = self.cfg.use_local_text_embeddings

            return MultiRefProcessorOutput4Text(
                device=self.device,
                prompts=prompts,
                **prompt_args,
            )

@dataclass
class MultiRefProcessorOutput4Text:
    text_embeddings_global: Optional[List[Float[Tensor, "B ..."]]] = None
    text_embeddings_local: Optional[List[Float[Tensor, "B ..."]]] = None
    uncond_text_embeddings_global: Optional[Float[Tensor, "B ..."]] = None
    uncond_text_embeddings_local: Optional[Float[Tensor, "B ..."]] = None
    
    # must have the following attributes
    use_local_text_embeddings: bool = False
    device: str = "cuda"

    appendable_attributes: list = field(
        default_factory=lambda: [
            "text_embeddings_global",
            "text_embeddings_local",
        ]
    )

    prompts: Optional[List[str]] = None

    def get_uncond_text_embeddings(self):
        # only the local embeddings are uncond
        batch_size = len(self.text_embeddings_global)

        uncond_text_embeddings_local = self.uncond_text_embeddings_local
        if isinstance(self.uncond_text_embeddings_local, List):
            uncond_text_embeddings_local = self.uncond_text_embeddings_local[0]

        return uncond_text_embeddings_local[None, :, :].repeat(
                batch_size, 1, 1
            ).to(self.device)
        
    def get_text_embeddings(
        self,
        **kwargs
    ):
        if "view_dependent_prompting" in kwargs and kwargs["view_dependent_prompting"]:
            raise NotImplementedError("View-dependent prompting is not supported for text embeddings")

        text_embeddings = torch.stack(
            self.text_embeddings_local,
            dim=0
        ).to(self.device)
        uncond_text_embeddings = self.get_uncond_text_embeddings()
        return torch.cat(
            [
                text_embeddings,
                uncond_text_embeddings,
            ],
            dim=0
        )

    def get_global_text_embeddings(
        self,
        use_local_text_embeddings: Optional[bool] = None,
    ):
        if use_local_text_embeddings is None:
            use_local_text_embeddings = self.use_local_text_embeddings

        if use_local_text_embeddings:
            return torch.stack(
                self.text_embeddings_local, 
                dim=0
            ).to(self.device)
        else:
            return torch.stack(
                self.text_embeddings_global, 
                dim=0
            ).to(self.device)

    

@dataclass
class MultiRefProcessorOutput4Image:
    image_latents: Optional[List[Float[Tensor, "B ..."]]] = None
    image_embeddings_global: Optional[List[Float[Tensor, "B ..."]]] = None
    image_embeddings_local: Optional[List[Float[Tensor, "B ..."]]] = None
    
    text_embeddings_global: Optional[List[Float[Tensor, "B ..."]]] = None
    text_embeddings_local: Optional[List[Float[Tensor, "B ..."]]] = None

    # must have the following attributes
    device: str = "cuda"

    appendable_attributes: list = field(
        default_factory=lambda: [
            "image_latents",
            "image_embeddings_global",
            "image_embeddings_local",
            "text_embeddings_global",
            "text_embeddings_local",
        ]
    )

    def get_uncond_image_encodings(self):
        return_dict = {}
        if self.image_latents is not None:
            return_dict["image_latents"] = torch.zeros_like(
                torch.stack(  
                    self.image_latents, 
                    dim=0
                )
            ).to(self.device)
        if self.image_embeddings_global is not None:
            return_dict["image_embeddings_global"] = torch.zeros_like(
                torch.stack(
                    self.image_embeddings_global, 
                    dim=0
                )
            ).to(self.device)
        if self.image_embeddings_local is not None:
            return_dict["image_embeddings_local"] = torch.zeros_like(
                torch.stack(
                    self.image_embeddings_local, 
                    dim=0
                )
            ).to(self.device)
        return return_dict


    def get_image_encodings(
        self,
    ):
        return_dict = {}
        if self.image_latents is not None:
            return_dict["image_latents"] = torch.stack(
                self.image_latents, 
                dim=0
            ).to(self.device)
        if self.image_embeddings_global is not None:
            return_dict["image_embeddings_global"] = torch.stack(
                self.image_embeddings_global, 
                dim=0
            ).to(self.device)
        if self.image_embeddings_local is not None:
            return_dict["image_embeddings_local"] = torch.stack(
                self.image_embeddings_local, 
                dim=0
            ).to(self.device)
        if self.text_embeddings_global is not None:
            return_dict["text_embeddings_global"] = torch.stack(
                self.text_embeddings_global, 
                dim=0
            ).to(self.device)
        if self.text_embeddings_local is not None:
            return_dict["text_embeddings_local"] = torch.stack(
                self.text_embeddings_local, 
                dim=0
            ).to(self.device)
        return return_dict