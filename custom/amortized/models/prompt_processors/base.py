import json
import os
from dataclasses import dataclass

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import *
from threestudio.utils.misc import barrier, cleanup

from tqdm import tqdm
from collections import OrderedDict

from threestudio.models.prompt_processors.base import (
    DirectionConfig, shift_azimuth_deg, PromptProcessorOutput,
    shifted_expotional_decay
)
from threestudio.utils.misc import get_rank
from torch.multiprocessing import Pool
import torch.distributed as dist

from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from .utils import _load_prompt_embedding, hash_prompt
from concurrent.futures import ThreadPoolExecutor,as_completed

class MultiPromptProcessor(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        prompt_library: str = "magic3d_prompt_library"
        prompt_library_dir: str = "load"
        prompt_library_format: str = "json"

        eval_prompt: Optional[str] = None
        eval_prompt_target: Optional[str] = None
        # # manually assigned view-dependent prompts
        # prompt_front: Optional[str] = None
        # prompt_side: Optional[str] = None
        # prompt_back: Optional[str] = None
        # prompt_overhead: Optional[str] = None
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"

        negative_prompt: str = ""
        overhead_threshold: float = 60.0
        front_threshold: float = 45.0
        back_threshold: float = 45.0
        view_dependent_prompt_front: bool = False
        use_cache: bool = True
        spawn: bool = False

        cache_dir: str = ".threestudio_cache/text_embeddings"  # FIXME: hard-coded path

        # perp neg
        use_perp_neg: bool = False
        # a*e(-b*r) + c
        # a * e(-b) + c = 0
        perp_neg_f_sb: Tuple[float, float, float] = (1, 0.5, -0.606)
        perp_neg_f_fsb: Tuple[float, float, float] = (1, 0.5, +0.967)
        perp_neg_f_fs: Tuple[float, float, float] = (
            4,
            0.5,
            -2.426,
        )  # f_fs(1) = 0, a, b > 0
        perp_neg_f_sf: Tuple[float, float, float] = (4, 0.5, -2.426)

        # # prompt debiasing
        use_prompt_debiasing: bool = False
        # pretrained_model_name_or_path_prompt_debiasing: str = "bert-base-uncased"
        # # index of words that can potentially be removed
        # prompt_debiasing_mask_ids: Optional[List[int]] = None
        use_local_text_embeddings: bool = False

    cfg: Config


    @rank_zero_only
    def configure_text_encoder(self) -> None:
        raise NotImplementedError

    @rank_zero_only
    def destroy_text_encoder(self) -> None:
        raise NotImplementedError
    
    def configure(self) -> None:

        rank = get_rank()
        num_gpus = torch.cuda.device_count()
        
        self._cache_dir = self.cfg.cache_dir
        
        # view-dependent text embeddings, same as in stable_diffusion_prompt_processor.py
        self.directions: List[DirectionConfig]

        # view-dependent text embeddings, same as in stable_diffusion_prompt_processor.py
        self.directions: List[DirectionConfig]
        if self.cfg.view_dependent_prompt_front:
            self.directions = [
                DirectionConfig(
                    "side",
                    lambda s: f"side view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: torch.ones_like(ele, dtype=torch.bool),
                ),
                DirectionConfig(
                    "front",
                    lambda s: f"front view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > -self.cfg.front_threshold
                    )
                    & (shift_azimuth_deg(azi) < self.cfg.front_threshold),
                ),
                DirectionConfig(
                    "back",
                    lambda s: f"backside view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > 180 - self.cfg.back_threshold
                    )
                    | (shift_azimuth_deg(azi) < -180 + self.cfg.back_threshold),
                ),
                DirectionConfig(
                    "overhead",
                    lambda s: f"overhead view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: ele > self.cfg.overhead_threshold,
                ),
            ]
        else:
            self.directions = [
                DirectionConfig(
                    "side",
                    lambda s: f"{s}, side view",
                    lambda s: s,
                    lambda ele, azi, dis: torch.ones_like(ele, dtype=torch.bool),
                ),
                DirectionConfig(
                    "front",
                    lambda s: f"{s}, front view",
                    lambda s: s,
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > -self.cfg.front_threshold
                    )
                    & (shift_azimuth_deg(azi) < self.cfg.front_threshold),
                ),
                DirectionConfig(
                    "back",
                    lambda s: f"{s}, back view",
                    lambda s: s,
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > 180 - self.cfg.back_threshold
                    )
                    | (shift_azimuth_deg(azi) < -180 + self.cfg.back_threshold),
                ),
                DirectionConfig(
                    "overhead",
                    lambda s: f"{s}, overhead view",
                    lambda s: s,
                    lambda ele, azi, dis: ele > self.cfg.overhead_threshold,
                ),
            ]

        self.direction2idx = {d.name: i for i, d in enumerate(self.directions)}

        if self.cfg.eval_prompt is None:
            # training
            path = os.path.join(
                self.cfg.prompt_library_dir, 
                self.cfg.prompt_library) \
                    + "." + self.cfg.prompt_library_format
            with open(path, "r") as f:
                prompt_library_json = json.load(f)
                all_prompts = []
                for split in prompt_library_json:
                    # each process only has a subset of the prompt library!
                    all_prompts += prompt_library_json[split][rank::num_gpus]

            self.prompt_library = []
            # # remove duplicates
            # for prompt in tqdm(all_prompts, desc="Removing duplicates from prompt library") \
            #         if get_rank() == 0 else all_prompts:
            #     if prompt not in self.prompt_library:
            #         self.prompt_library.append(prompt)
            self.prompt_library = all_prompts
        else:
            # evaluation
            self.prompt_library = [self.cfg.eval_prompt]
            if self.cfg.eval_prompt_target is not None:
                self.prompt_library.append(self.cfg.eval_prompt_target)


        # use provided negative prompt
        self.negative_prompt = self.cfg.negative_prompt
        threestudio.info(
            f"Using prompt library located in [{self.cfg.prompt_library}] and negative prompt [{self.negative_prompt}]"
        )        

        # view-dependent prompting
        self.prompts_vd = OrderedDict()
        if self.cfg.use_prompt_debiasing:
            raise NotImplementedError("Prompt debiasing is not implemented yet")
        else:
            for prompt in self.prompt_library:
                self.prompts_vd[prompt] = [d.prompt(prompt) for d in self.directions]

        self.negative_prompts_vd = [
            d.negative_prompt(self.negative_prompt) for d in self.directions
        ]

        # extract text embeddings and save to disk
        self.prepare_text_embeddings()

        self.text_embeddings_vd = OrderedDict()
        self.global_text_embeddings = OrderedDict()
        self.local_text_embeddings = OrderedDict()
        
        # load text embeddings from disk to memory, only allow one gpu to load
        self.load_text_embeddings()

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir, tokenizer = None, text_encoder = None):
        raise NotImplementedError
    
    #@rank_zero_only, deprecated when each process has its own cache
    def prepare_text_embeddings(self):
        os.makedirs(self._cache_dir, exist_ok=True)

        prompts_vds = []
        for prompt, prompt_vds in self.prompts_vd.items():
            prompts_vds.extend(prompt_vds)

        all_prompts = (
            self.prompt_library
            + [self.negative_prompt]
            + prompts_vds
            + self.negative_prompts_vd
        )
        prompts_to_process = []
        for prompt in all_prompts:
            if self.cfg.use_cache:
                # some text embeddings are already in cache
                # do not process them
                cache_path_global = os.path.join(
                    self._cache_dir,
                    f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt, 'global')}.pt",
                )

                cache_path_local = os.path.join(    
                    self._cache_dir,
                    f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt, 'local')}.pt",
                )

                if os.path.exists(cache_path_global) and os.path.exists(cache_path_local):
                    threestudio.debug(
                        f"Text embeddings for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] are already in cache, skip processing."
                    )
                    continue
            prompts_to_process.append(prompt)

        if len(prompts_to_process) > 0:

            # load tokenizer and text encoder for multiprocessing
            from transformers import AutoTokenizer, CLIPTextModel
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
            )
            text_encoder = CLIPTextModel.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="text_encoder",
                device_map="auto",
            )

            if self.cfg.spawn:
                threestudio.info(f"Spawning {len(prompts_to_process)} processes to process prompts.")
                # multiprocessing
                ctx = mp.get_context("spawn")
                subprocess = ctx.Process(
                    target=self.spawn_func,
                    args=(
                        self.cfg.pretrained_model_name_or_path,
                        prompts_to_process,
                        self._cache_dir,
                    ),
                )

                subprocess.start()
                subprocess.join()
                threestudio.info(f"Finished processing prompts.")

            else:
                # single process
                from tqdm import tqdm
                for prompt in tqdm(prompts_to_process, desc="Processing prompts"):
                    self.spawn_func(
                        self.cfg.pretrained_model_name_or_path,
                        prompt,
                        self._cache_dir,
                        tokenizer, 
                        text_encoder,
                    )


            del tokenizer
            del text_encoder
            cleanup()


    def load_text_embeddings(self):
        # synchronize, to ensure the text embeddings have been computed and saved to cache
        barrier()

        # check if is distributed
        num_gpus = torch.cuda.device_count()
        is_distributed = num_gpus > 1
        rank = get_rank()

        prompt, prompt_vds = self.prompts_vd.keys(), self.prompts_vd.values()
        # add negative prompts
        prompt = list(prompt) + [self.negative_prompt]
        prompt_vds = list(prompt_vds) + [self.negative_prompts_vd]

        if is_distributed:
            # Use ThreadPoolExecutor to parallelize the loading of embeddings
            with ThreadPoolExecutor(max_workers=4) as executor: # hard-coded max_workers
                # Use as_completed to retrieve the results as they are completed
                futures = {
                        executor.submit(_load_prompt_embedding, args): args for args \
                            in zip(
                                prompt,
                                prompt_vds,
                                cycle([self._cache_dir]),
                                cycle([self.cfg.pretrained_model_name_or_path]),
                            )
                    }
                for future in tqdm(as_completed(futures), desc="Loading text embeddings in {}".format(rank), total=len(prompt)):
                    data = future.result()
                    p, global_text_embeddings, local_text_embeddings, text_embeddings_vd = data
                    self.global_text_embeddings[p] = global_text_embeddings
                    self.local_text_embeddings[p] = local_text_embeddings
                    self.text_embeddings_vd[p] = text_embeddings_vd
        else:
            # for debugging and single-gpu, single process
            for data in tqdm(
                map(
                    _load_prompt_embedding,
                    zip(
                        prompt,  # [rank::num_gpus] is to split the list into num_gpus parts, and get the rank-th part
                        prompt_vds,
                        cycle([self._cache_dir]),
                        cycle([self.cfg.pretrained_model_name_or_path]),
                    ),
                ),
                desc="Loading text embeddings for GPU: {}".format(rank)
                    if is_distributed else "Loading text embeddings",
                total=len(prompt),
            ):
                p, global_text_embeddings, local_text_embeddings, text_embeddings_vd = data
                self.global_text_embeddings[p] = global_text_embeddings
                self.local_text_embeddings[p] = local_text_embeddings
                self.text_embeddings_vd[p] = text_embeddings_vd



        barrier()
        threestudio.debug(f"Loaded text embeddings.")
        

        
    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B ..."], Float[Tensor, "B ..."]]:
        raise NotImplementedError
    
    def __call__(
        self,
        prompt: Union[str, List[str]],
    ) -> PromptProcessorOutput:
        if isinstance(prompt, str):
            prompt = [prompt]
        
        # check if the prompt is in the prompt library
        for p in prompt:
            if p not in self.prompt_library:
                raise ValueError(f"Prompt [{p}] is not in the prompt library.")
            
        obj = MultiPromptProcessorOutput(
                global_text_embeddings=[self.global_text_embeddings[p] for p in prompt],
                local_text_embeddings=[self.local_text_embeddings[p] for p in prompt],
                uncond_text_embeddings=self.local_text_embeddings[self.negative_prompt],
                text_embeddings_vd=[self.text_embeddings_vd[p] for p in prompt],
                uncond_text_embeddings_vd=self.text_embeddings_vd[self.negative_prompt],
                directions=self.directions,
                direction2idx=self.direction2idx,
                use_perp_neg=self.cfg.use_perp_neg,
                perp_neg_f_sb=self.cfg.perp_neg_f_sb,
                perp_neg_f_fsb=self.cfg.perp_neg_f_fsb,
                perp_neg_f_fs=self.cfg.perp_neg_f_fs,
                perp_neg_f_sf=self.cfg.perp_neg_f_sf,
                device=self.device,
                use_local_text_embeddings=self.cfg.use_local_text_embeddings
            )
        return obj
        
@dataclass
class MultiPromptProcessorOutput:
    global_text_embeddings: List[Float[Tensor, "B ..."]]
    local_text_embeddings: List[Float[Tensor, "B ..."]]
    uncond_text_embeddings: Float[Tensor, "B ..."]
    text_embeddings_vd: List[Float[Tensor, "D B ..."]]
    uncond_text_embeddings_vd: Float[Tensor, "B ..."]
    directions: List[DirectionConfig]
    direction2idx: Dict[str, int]
    use_perp_neg: bool
    perp_neg_f_sb: Tuple[float, float, float]
    perp_neg_f_fsb: Tuple[float, float, float]
    perp_neg_f_fs: Tuple[float, float, float]
    perp_neg_f_sf: Tuple[float, float, float]
    device: str = "cuda"
    use_local_text_embeddings: bool = False

    def get_global_text_embeddings(
        self,
    ):
        if self.use_local_text_embeddings:
            return torch.stack(self.local_text_embeddings, dim=0).to(self.device)
        else:
            return torch.stack(self.global_text_embeddings, dim=0).to(self.device)

    def get_text_embeddings(
        self, 
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        view_dependent_prompting: bool = True,
    ) -> Float[Tensor, "BB N Nf"]:
        batch_size = len(self.global_text_embeddings)

        if view_dependent_prompting:
            # Get direction
            direction_idx = torch.zeros_like(elevation, dtype=torch.long).to('cpu')
            for d in self.directions:
                direction_idx[
                    d.condition(elevation, azimuth, camera_distances)
                ] = self.direction2idx[d.name]

            # Get text embeddings
            text_embeddings = torch.stack(
                [
                    self.text_embeddings_vd[i][direction_idx[i]]
                    for i in range(batch_size)
                ],
                dim=0,
            )
            uncond_text_embeddings = self.uncond_text_embeddings_vd[direction_idx]
        else:
            text_embeddings = torch.stack(
                [self.local_text_embeddings[i] for i in range(batch_size)], dim=0
            )
            uncond_text_embeddings = self.uncond_text_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1
            )

        # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
        return torch.cat([text_embeddings, uncond_text_embeddings], dim=0).to(self.device)

    def get_text_embeddings_perp_neg(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        view_dependent_prompting: bool = True,
        guidance_scale_neg: Optional[float] = None,
    ) -> Tuple[Float[Tensor, "BB N Nf"], Float[Tensor, "B 2"]]:
        assert (
            view_dependent_prompting
        ), "Perp-Neg only works with view-dependent prompting"

        batch_size = len(self.global_text_embeddings)

        if guidance_scale_neg is None:
            guidance_scale_neg = -1

        direction_idx = torch.zeros_like(elevation, dtype=torch.long).to('cpu')
        for d in self.directions:
            direction_idx[
                d.condition(elevation, azimuth, camera_distances)
            ] = self.direction2idx[d.name]
        
        # 0 - side view
        # 1 - front view
        # 2 - back view
        # 3 - overhead view

        pos_text_embeddings = []
        neg_text_embeddings = []
        neg_guidance_weights = []
        uncond_text_embeddings = []


        # similar to get_text_embeddings_perp_neg in stable_diffusion_prompt_processor.py
        for batch_idx in range(batch_size):
            # get direction
            idx = direction_idx[batch_idx].to('cpu')
            ele = elevation[batch_idx].to('cpu')
            azi = azimuth[batch_idx].to('cpu')
            dis = camera_distances[batch_idx].to('cpu')

            # get text embeddings
            side_emb = self.text_embeddings_vd[batch_idx][0]
            front_emb = self.text_embeddings_vd[batch_idx][1]
            back_emb = self.text_embeddings_vd[batch_idx][2]
            overhead_emb = self.text_embeddings_vd[batch_idx][3]

            # the following code is similar to stable_diffusion_prompt_processor.py
            azi = shift_azimuth_deg(azi)  # to (-180, 180)
            uncond_text_embeddings.append(
                self.uncond_text_embeddings_vd[idx]
            )  # should be ""
            if idx.item() == 3:  # overhead view
                pos_text_embeddings += [overhead_emb]
                # dummy
                neg_text_embeddings += [
                    self.uncond_text_embeddings_vd[idx],
                    self.uncond_text_embeddings_vd[idx],
                ]
                neg_guidance_weights += [0.0, 0.0]
            else:  # interpolating views
                if torch.abs(azi) < 90:
                    # front-side interpolation
                    # 0 - complete side, 1 - complete front
                    r_inter = 1 - torch.abs(azi) / 90
                    pos_text_embeddings.append(
                        r_inter * front_emb + (1 - r_inter) * side_emb
                    )
                    neg_text_embeddings += [front_emb, side_emb]
                    neg_guidance_weights += [
                        shifted_expotional_decay(*self.perp_neg_f_fs, r_inter) * guidance_scale_neg,
                        shifted_expotional_decay(*self.perp_neg_f_sf, 1 - r_inter) * guidance_scale_neg,
                    ]
                else:
                    # side-back interpolation
                    # 0 - complete back, 1 - complete side
                    r_inter = 2.0 - torch.abs(azi) / 90
                    pos_text_embeddings.append(
                        r_inter * side_emb + (1 - r_inter) * back_emb
                    )
                    neg_text_embeddings += [side_emb, front_emb]
                    neg_guidance_weights += [
                        shifted_expotional_decay(*self.perp_neg_f_sb, r_inter) * guidance_scale_neg,
                        shifted_expotional_decay(*self.perp_neg_f_fsb, r_inter) * guidance_scale_neg,
                    ]

        text_embeddings = torch.cat(
            [
                torch.stack(pos_text_embeddings, dim=0),
                torch.stack(uncond_text_embeddings, dim=0),
                torch.stack(neg_text_embeddings, dim=0),
            ],
            dim=0,
        )

        return text_embeddings.to(self.device), torch.as_tensor(
            neg_guidance_weights, device=elevation.device
        ).reshape(batch_size, 2).to(self.device)