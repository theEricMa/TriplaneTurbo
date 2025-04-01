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

from functools import partial

from threestudio.models.prompt_processors.base import (
    DirectionConfig, shift_azimuth_deg, PromptProcessorOutput,
    shifted_expotional_decay
)
from threestudio.utils.misc import get_rank

from itertools import cycle
from .utils import _load_prompt_embedding, hash_prompt

##############################################
# the following fucntions are warpped up
# so that they can be used in the multiprocessing
def format_view(s: str, view: str) -> str:
    return f"{view} view of {s}"

def format_view_v2(s: str, view: str) -> str:
    return f"{s}, {view} view"

def dummy_map_fn(x):
    return x

def is_within_back_view(ele, azi, dis, back_threshold):
    shifted_azi = shift_azimuth_deg(azi)
    return (shifted_azi > 180 - back_threshold) | (shifted_azi < -180 + back_threshold)

def is_within_front_view(ele, azi, dis, front_threshold):
    shifted_azi = shift_azimuth_deg(azi)
    return (shifted_azi > -front_threshold) & (shifted_azi < front_threshold)

def is_above_overhead_threshold(ele, azi, dis, overhead_threshold):
    return ele > overhead_threshold

def is_within_side_view(ele, azi, dis):
    return torch.ones_like(ele, dtype=torch.bool)
##############################################

class MultiPromptProcessor(BaseObject):
    @dataclass
    class Config(BaseObject.Config):

        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"

        negative_prompt: str = ""
        negative_prompt_2nd: Optional[str] = None

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
        perp_neg_f_fs: Tuple[float, float, float] = (4, 0.5, -2.426, )  # f_fs(1) = 0, a, b > 0
        perp_neg_f_sf: Tuple[float, float, float] = (4, 0.5, -2.426)

        use_local_text_embeddings: bool = False
        use_view_dependent_text_embeddings: Optional[List[str]] = None #field(default_factory=lambda: [])



    cfg: Config

    def __reduce__(self) -> str | tuple[Any, ...]:
        return (self.__class__, (self.configure,))
    
    def configure(self) -> None:
        
        self._cache_dir = self.cfg.cache_dir
        
        # view-dependent text embeddings, same as in stable_diffusion_prompt_processor.py
        self.directions: List[DirectionConfig]

        # view-dependent text embeddings, same as in stable_diffusion_prompt_processor.py
        self.directions: List[DirectionConfig]
        if self.cfg.view_dependent_prompt_front:
            self.directions = [
                DirectionConfig(
                    "side",
                    partial(format_view, view="side"),
                    dummy_map_fn,
                    partial(is_within_side_view),
                ),
                DirectionConfig(
                    "front",
                    partial(format_view, view="front"),
                    dummy_map_fn,
                    partial(is_within_front_view, front_threshold=self.cfg.front_threshold),

                ),
                DirectionConfig(
                    "back",
                    partial(format_view, view="back"),
                    dummy_map_fn,
                    partial(is_within_back_view, back_threshold=self.cfg.back_threshold),

                ),
                DirectionConfig(
                    "overhead",
                    partial(format_view, view="overhead"),
                    dummy_map_fn,
                    partial(is_above_overhead_threshold, overhead_threshold=self.cfg.overhead_threshold),
                ),
            ]
        else:
            self.directions = [
                DirectionConfig(
                    "side",
                    partial(format_view_v2, view="side"),
                    dummy_map_fn,
                    partial(is_within_side_view),
                ),
                DirectionConfig(
                    "front",
                    partial(format_view_v2, view="front"),
                    dummy_map_fn,
                    partial(is_within_front_view, front_threshold=self.cfg.front_threshold),
                ),
                DirectionConfig(
                    "back",
                    partial(format_view_v2, view="back"),
                    dummy_map_fn,
                    partial(is_within_back_view, back_threshold=self.cfg.back_threshold),
                ),
                DirectionConfig(
                    "overhead",
                    partial(format_view_v2, view="overhead"),
                    dummy_map_fn,
                    partial(is_above_overhead_threshold, overhead_threshold=self.cfg.overhead_threshold),
                ),
            ]

        self.direction2idx = {d.name: i for i, d in enumerate(self.directions)}



        # use provided negative prompt
        self.negative_prompt = self.cfg.negative_prompt
        self.negative_prompts_vd = self._view_dependent_text_embeddings(self.negative_prompt)
        # 2nd negative prompt
        if self.cfg.negative_prompt_2nd is not None:
            self.negative_prompt_2nd = self.cfg.negative_prompt_2nd
            self.negative_prompts_vd_2nd = self._view_dependent_text_embeddings(self.negative_prompt_2nd)

    @staticmethod
    def func(pretrained_model_name_or_path, prompts, cache_dir, tokenizer = None, text_encoder = None):
        raise NotImplementedError
    
    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir):
        raise NotImplementedError
    
    def _view_dependent_text_embeddings(self, prompt: str) -> List[str]:
        return [d.prompt(prompt) for d in self.directions]

    #@rank_zero_only, deprecated when each process has its own cache
    def prepare_text_embeddings(self, prompt_library: List[str]) -> None:

        os.makedirs(self._cache_dir, exist_ok=True)

        rank = get_rank()
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            # each process only has a subset of the prompt library!
            prompt_library = prompt_library[rank::num_gpus]

        # for each prompt in the prompt library
        # we need to process also their view-dependent prompts
        prompts_vd = []
        for prompt in prompt_library:
            prompts_vd += self._view_dependent_text_embeddings(prompt)

        # collect all prompts
        all_prompts = (
            prompt_library
            + prompts_vd
        )

        # if (self.cfg.gpu_split and rank == 0) or not self.cfg.gpu_split:
        if rank == 0:
            # rank 0 process will also process the negative prompts
            all_prompts += [self.negative_prompt]
            all_prompts += self.negative_prompts_vd

            # 2nd negative prompt
            if self.negative_prompt_2nd is not None:
                all_prompts += [self.negative_prompt_2nd]
                all_prompts += self.negative_prompts_vd_2nd


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

            if self.cfg.spawn:
                # # try 1-st approach
                # threestudio.info(f"Spawning {len(prompts_to_process)} processes to process prompts.")
                # # multiprocessing
                # ctx = mp.get_context("spawn")
                # subprocess = ctx.Process(
                #     target=self.spawn_func,
                #     args=(
                #         self.cfg.pretrained_model_name_or_path,
                #         prompts_to_process,
                #         self._cache_dir,
                #     ),
                # )

                # subprocess.start()
                # subprocess.join()
                # threestudio.info(f"Finished processing prompts.")

                # # try 2-nd approach
                # from torch.multiprocessing import Pool, set_start_method
                # set_start_method("spawn")
                # threestudio.info(f"Spawning {len(prompts_to_process)} processes to process prompts.")
                # num_workers = 0  # hard-coded number of processes
                # pool = Pool(num_workers)
                # args_list = [
                #     (
                #         self.cfg.pretrained_model_name_or_path,
                #         prompts_to_process[idx_worker::num_workers],
                #         self._cache_dir,
                #     ) for idx_worker in range(num_workers)
                # ]

                # for _ in pool.imap_unordered(self.spawn_func, args_list):
                #     pass
                # threestudio.info(f"Finished processing prompts.")

                # try 3-rd approach
                self.spawn_func(
                    (
                        self.cfg.pretrained_model_name_or_path,
                        prompts_to_process,
                        self._cache_dir,
                    )
                )

            else:

                # load tokenizer and text encoder in the main process
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

                for p in text_encoder.parameters():
                    p.requires_grad_(False)

                # single process
                from tqdm import tqdm
                for prompt in tqdm(prompts_to_process, desc="Processing prompts"):
                    self.func(
                        self.cfg.pretrained_model_name_or_path,
                        prompt,
                        self._cache_dir,
                        tokenizer, 
                        text_encoder,
                    )

                # no need to keep tokenizer and text_encoder in memory
                del tokenizer
                del text_encoder
                cleanup()

    def load_text_embeddings(self, prompt_batch: List[str]) -> None:

        prompt = prompt_batch.copy()

        prompt_vds = []
        for p in prompt:
            prompt_vds.append(
                self._view_dependent_text_embeddings(p)
            )

        # add negative prompts
        prompt += [self.negative_prompt]
        prompt_vds.append(self.negative_prompts_vd)
        # 2nd negative prompt
        if self.negative_prompt_2nd is not None:
            prompt += [self.negative_prompt_2nd]
            prompt_vds.append(self.negative_prompts_vd_2nd)

        global_text_embeddings_dict = {}
        local_text_embeddings_dict = {}
        text_embeddings_vd_dict = {}

        # for debugging and single-gpu, single process
        for data in map(
                _load_prompt_embedding,
                zip(
                    prompt,  # [rank::num_gpus] is to split the list into num_gpus parts, and get the rank-th part
                    prompt_vds,
                    cycle([self._cache_dir]),
                    cycle([self.cfg.pretrained_model_name_or_path]),
                ),
            ):

            p, global_text_embeddings, local_text_embeddings, text_embeddings_vd = data
            global_text_embeddings_dict[p] = global_text_embeddings
            local_text_embeddings_dict[p] = local_text_embeddings
            text_embeddings_vd_dict[p] = text_embeddings_vd

        return global_text_embeddings_dict, local_text_embeddings_dict, text_embeddings_vd_dict
 
    def __call__(
        self,
        prompt: Union[str, List[str]],
    ) -> PromptProcessorOutput:
        if isinstance(prompt, str):
            prompt = [prompt]
        
        global_text_embeddings, local_text_embeddings, text_embeddings_vd = self.load_text_embeddings(prompt)
        
        prompt_args = {
            "global_text_embeddings": [global_text_embeddings[p ] for p in prompt],
            "local_text_embeddings": [local_text_embeddings[p] for p in prompt],
            "uncond_text_embeddings": local_text_embeddings[self.negative_prompt],
            "text_embeddings_vd": [text_embeddings_vd[p] for p in prompt],
            "uncond_text_embeddings_vd": text_embeddings_vd[self.negative_prompt],
        }
        # 2nd negative prompt
        if self.negative_prompt_2nd is not None:
            prompt_args["uncond_text_embeddings_2nd"] = local_text_embeddings[self.negative_prompt_2nd]
            prompt_args["uncond_text_embeddings_vd_2nd"] = text_embeddings_vd[self.negative_prompt_2nd]

        direction_args = {
            "directions": self.directions,
            "direction2idx": self.direction2idx,
            "use_perp_neg": self.cfg.use_perp_neg,
            "perp_neg_f_sb": self.cfg.perp_neg_f_sb,
            "perp_neg_f_fsb": self.cfg.perp_neg_f_fsb,
            "perp_neg_f_fs": self.cfg.perp_neg_f_fs,
            "perp_neg_f_sf": self.cfg.perp_neg_f_sf,
        }
        obj = MultiPromptProcessorOutput(
            device=self.device,
            use_local_text_embeddings=self.cfg.use_local_text_embeddings,
            use_view_dependent_text_embeddings=self.cfg.use_view_dependent_text_embeddings,
            **prompt_args,
            **direction_args,
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
    use_view_dependent_text_embeddings: Optional[List[str]] = None
    uncond_text_embeddings_2nd: Optional[Float[Tensor, "B ..."]] = None
    uncond_text_embeddings_vd_2nd: Optional[Float[Tensor, "B ..."]] = None

    def get_uncond_text_embeddings(self):
        batch_size = len(self.global_text_embeddings)
        if self.use_view_dependent_text_embeddings is not None:
            return self.uncond_text_embeddings[None, None, :, :].repeat(
                batch_size, len(self.use_view_dependent_text_embeddings), 1, 1
            ).to(self.device)
        else:
            return self.uncond_text_embeddings[None, :, :].repeat(
                batch_size, 1, 1
            ).to(self.device)

    def get_global_text_embeddings(
        self,
        use_local_text_embeddings: Optional[bool] = None,
    ):

        if use_local_text_embeddings is None:
            use_local_text_embeddings = self.use_local_text_embeddings

        if use_local_text_embeddings:
            if not self.use_view_dependent_text_embeddings:
                return torch.stack(self.local_text_embeddings, dim=0).to(self.device)
            else:
                feature_list_batch = []
                for prompt_vd, prompt in zip(self.text_embeddings_vd, self.local_text_embeddings):
                    feature_list = []
                    for view in self.use_view_dependent_text_embeddings:
                        assert view in ["front", "side", "back", "overhead", "none", "uncond_1st"], f"Invalid view-dependent text embeddings: {view}"
                        if view == "none": # special case for not using view-dependent text embeddings
                            feature_list.append(
                                prompt
                            )
                        elif view == "uncond_1st":
                            feature_list.append(
                                self.uncond_text_embeddings
                            )
                        else:
                            direction_idx = self.direction2idx[view]
                            feature_list.append(
                                prompt_vd[direction_idx]
                            )
                    feature_list_batch.append(
                        torch.stack(
                            feature_list, 
                            dim=0
                        )
                    )
                return torch.stack(feature_list_batch, dim=0).to(self.device)
        else:
            return torch.stack(self.global_text_embeddings, dim=0).to(self.device)
        
            # if not self.use_view_dependent_text_embeddings:
            #     return torch.stack(self.global_text_embeddings, dim=0).to(self.device)
            # else:
            #     raise NotImplementedError("View-dependent text embeddings are not supported yet.")

    def get_text_embeddings(
        self, 
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        view_dependent_prompting: bool = True,
        use_2nd_uncond: bool = False,
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
            # 2nd negative prompt
            if use_2nd_uncond:
                uncond_text_embeddings = self.uncond_text_embeddings_vd_2nd[direction_idx]
        else:
            text_embeddings = torch.stack(
                [self.local_text_embeddings[i] for i in range(batch_size)], dim=0
            )
            uncond_text_embeddings = self.uncond_text_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            # 2nd negative prompt
            if use_2nd_uncond:
                uncond_text_embeddings = self.uncond_text_embeddings_2nd.unsqueeze(0).expand(
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
        use_2nd_uncond: bool = False,
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
                self.uncond_text_embeddings_vd[idx] if not use_2nd_uncond else self.uncond_text_embeddings_vd_2nd[idx]
            )  # should be ""
            if idx.item() == 3:  # overhead view
                pos_text_embeddings += [overhead_emb]
                # dummy
                neg_text_embeddings += [
                    self.uncond_text_embeddings_vd[idx] if not use_2nd_uncond else self.uncond_text_embeddings_vd_2nd[idx],
                    self.uncond_text_embeddings_vd[idx] if not use_2nd_uncond else self.uncond_text_embeddings_vd_2nd[idx],
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