# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Gradient collection entry point (simplified).

Flow:
  1. sample one big batch of N prompts from the dataset (N = 32 * 20)
  2. rollout once, compute reward / old_log_probs / (rollout-correction) / advantage
  3. split into 20 chunks of 32 prompts and run forward+backward per chunk,
     dumping a single merged gradient file per chunk. The model is never updated.

Output:
  ``trainer.grad_save_dir/batch_{i:04d}.pt`` -- dict[str, torch.Tensor] in HF
  parameter names, full DP/TP/PP-merged gradient stored as bf16, only on global
  rank 0.
"""

import os
import socket
import uuid

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer
from verl import DataProto
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import compute_advantage, compute_response_mask
from verl.trainer.ppo.reward import compute_reward, load_reward_manager
from verl.utils.device import get_torch_device
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    offload_megatron_model_to_cpu,
    per_tensor_generator,
)
from verl.workers.megatron_workers import AsyncActorRolloutRefWorker as _BaseAsyncWorker


class GradCollectActorRolloutRefWorker(_BaseAsyncWorker):
    """Actor worker that performs forward+backward, merges grads, dumps a single file."""

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_and_save_grad(self, data: DataProto):
        assert self._is_actor

        save_dir = data.meta_info["grad_save_dir"]
        step = data.meta_info["grad_step"]

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)

        for chunk in self.actor_module:
            chunk.zero_grad_buffer()

        micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        max_token_len = None
        if self.actor.config.use_dynamic_bsz:
            max_token_len = (
                self.actor.config.ppo_max_token_len_per_gpu
                * self.actor.config.megatron.context_parallel_size
            )

        # Megatron's forward_backward_func runs finalize_model_grads_func at the
        # end -> param.main_grad is already DP-reduced after this call.
        self.actor.forward_backward_batch(
            data,
            calculate_entropy=self.actor.config.entropy_coeff != 0,
            use_dynamic_bsz=self.actor.config.use_dynamic_bsz,
            micro_batch_size=micro_batch_size,
            max_token_len=max_token_len,
            mini_batch_size=self.actor.config.ppo_mini_batch_size,
        )

        # Reuse verl's per_tensor_generator (PP broadcast + TP/EP all-gather +
        # HF name conversion) on grads by temporarily swapping param.data with
        # param.main_grad.
        swapped = []
        for chunk in self.actor_module:
            inner = chunk.module if hasattr(chunk, "module") else chunk
            for _, param in inner.named_parameters():
                if not param.requires_grad:
                    continue
                grad = getattr(param, "main_grad", None)
                if grad is None:
                    grad = param.grad
                if grad is None:
                    continue
                swapped.append((param, param.data))
                param.data = grad

        try:
            gen = per_tensor_generator(
                self.actor.actor_module,
                self.actor_model_config,
                self.weight_converter,
                self.tf_config,
                self.layer_name_mapping,
            )
            merged: dict[str, torch.Tensor] = {}
            for name, tensor in gen:
                merged[name] = tensor.detach().to(torch.bfloat16).cpu().clone()
        finally:
            for param, original_data in swapped:
                param.data = original_data

        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        out_path = os.path.join(save_dir, f"batch_{step:04d}.pt")
        if global_rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(merged, out_path)

        del merged
        for chunk in self.actor_module:
            chunk.zero_grad_buffer()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        get_torch_device().empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return DataProto(meta_info={"saved_path": out_path if global_rank == 0 else None})


class GradCollectTrainer(RayDAPOTrainer):
    """Loop ``num_grad_batches`` times: rollout one small batch, compute grad,
    dump merged grad file. No optimizer step."""

    def fit(self):
        self.global_steps = 0
        self.gen_steps = 0
        self._load_checkpoint()

        num_batches = int(self.config.trainer.num_grad_batches)
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        grad_save_dir = self.config.trainer.grad_save_dir
        os.makedirs(grad_save_dir, exist_ok=True)

        from verl.trainer.ppo.rollout_corr_helper import (
            compute_rollout_correction_and_add_to_batch,
        )

        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

        pbar = tqdm(total=num_batches, desc="Grad collection")
        collected = 0
        for _epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if collected >= num_batches:
                    break

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)

                # ---- rollout ----
                gen_batch = self._get_gen_batch(new_batch)
                gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)
                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                gen_batch_output.meta_info.pop("timing", None)

                new_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                )
                new_batch = new_batch.repeat(repeat_times=rollout_n, interleave=True)
                new_batch = new_batch.union(gen_batch_output)

                # ---- reward ----
                reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)
                new_batch.batch["token_level_scores"] = reward_tensor
                new_batch.batch["token_level_rewards"] = reward_tensor
                if reward_extra_infos_dict:
                    new_batch.non_tensor_batch.update(
                        {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                    )

                # ---- response_mask + old_log_probs (needed for PPO loss in backward) ----
                new_batch.batch["response_mask"] = compute_response_mask(new_batch)
                new_batch.meta_info["global_token_num"] = torch.sum(
                    new_batch.batch["attention_mask"], dim=-1
                ).tolist()
                old_log_prob = self.actor_rollout_wg.compute_log_prob(new_batch)
                old_log_prob.batch.pop("entropys", None)
                new_batch = new_batch.union(old_log_prob)

                # ---- rollout correction (TIS), if enabled ----
                if rollout_corr_config is not None and "rollout_log_probs" in new_batch.batch:
                    new_batch, _ = compute_rollout_correction_and_add_to_batch(
                        new_batch, rollout_corr_config
                    )

                # ---- advantages ----
                new_batch = compute_advantage(
                    new_batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=rollout_n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                )

                # ---- forward+backward+merge grad+dump ----
                new_batch.meta_info["grad_save_dir"] = grad_save_dir
                new_batch.meta_info["grad_step"] = collected
                self.actor_rollout_wg.compute_and_save_grad(new_batch)

                # ---- Save raw gradient directly (no Welford for single batch) ----
                grad_path = os.path.join(grad_save_dir, f"batch_{collected:04d}.pt")
                raw_grad_path = os.path.join(grad_save_dir, "grad_raw.pt")

                # Rename the batch file to grad_raw.pt
                if os.path.exists(grad_path):
                    os.rename(grad_path, raw_grad_path)
                    print(f"[grad_collect] saved raw gradient to {raw_grad_path}", flush=True)

                collected += 1
                pbar.update(1)

            if collected >= num_batches:
                break

        pbar.close()
        print(f"[grad_collect] collected {collected} batch(es) -> {grad_save_dir}/grad_raw.pt")


@hydra.main(config_path="../../verl/recipe/dapo/config", config_name="dapo_megatron_trainer", version_base=None)
def main(config):
    run(config)


def run(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    try:
        runner = TaskRunner.remote()
        ray.get(runner.run.remote(config))
    finally:
        if ray.is_initialized():
            ray.shutdown()


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from verl.utils.fs import copy_to_local
        from verl.utils import hf_processor, hf_tokenizer
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.workers.megatron_workers import CriticWorker

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        assert config.actor_rollout_ref.actor.strategy == "megatron"
        ray_worker_group_cls = RayWorkerGroup

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(GradCollectActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        reward_fn = load_reward_manager(
            config,
            tokenizer,
            0,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )
        val_reward_fn = load_reward_manager(
            config,
            tokenizer,
            1,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        trainer = GradCollectTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
