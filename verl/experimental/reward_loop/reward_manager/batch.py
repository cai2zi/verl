# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import inspect
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


@register("batch")
class BatchRewardManager(RewardManagerBase):
    """Reward manager that calls a custom reward function once per DataProto batch."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    @staticmethod
    def _to_reward_output(result: Any) -> dict:
        reward_extra_info = {}
        if isinstance(result, dict):
            score = result["score"]
            reward_extra_info.update(result)
        else:
            score = result
            reward_extra_info["acc"] = score
        return {"reward_score": score, "reward_extra_info": reward_extra_info}

    def _extra_reward_kwargs(self) -> dict:
        if self.reward_router_address is None:
            return {}
        return {
            "reward_router_address": self.reward_router_address,
            "reward_model_tokenizer": self.reward_model_tokenizer,
        }

    async def _prepare_batch_inputs(self, data: DataProto):
        data_sources = []
        ground_truths = []
        extra_infos = []
        solution_strs = []

        for i in range(len(data)):
            data_item = data[i]
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]

            data_sources.append(data_item.non_tensor_batch["data_source"])
            ground_truths.append(data_item.non_tensor_batch["reward_model"]["ground_truth"])

            extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}) or {})
            tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
            if tool_extra_fields is not None:
                extra_info.update(tool_extra_fields.items())

            if "uid" in data_item.non_tensor_batch:
                extra_info.setdefault("uid", str(data_item.non_tensor_batch["uid"]))
            if "global_steps" in data.meta_info:
                extra_info.setdefault("global_steps", data.meta_info["global_steps"])

            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores
            extra_infos.append(extra_info)

            solution_strs.append(self.tokenizer.decode(valid_response_ids, skip_special_tokens=True))

        return data_sources, solution_strs, ground_truths, extra_infos

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        return (await self.run_batch(data))[0]

    async def run_batch(self, data: DataProto) -> list[dict]:
        if len(data) == 0:
            return []

        data_sources, solution_strs, ground_truths, extra_infos = await self._prepare_batch_inputs(data)
        extra_reward_kwargs = self._extra_reward_kwargs()
        loop = asyncio.get_running_loop()
        if self.is_async_reward_score:
            results = await self.compute_score(
                data_sources=data_sources,
                solution_strs=solution_strs,
                ground_truths=ground_truths,
                extra_infos=extra_infos,
                **extra_reward_kwargs,
            )
        else:
            results = await loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_sources=data_sources,
                    solution_strs=solution_strs,
                    ground_truths=ground_truths,
                    extra_infos=extra_infos,
                    **extra_reward_kwargs,
                ),
            )
        if inspect.isawaitable(results):
            results = await results

        if not isinstance(results, (list, tuple)):
            if len(data) == 1:
                results = [results]
            else:
                raise ValueError(
                    "Batched reward function must return one result per data item, "
                    f"but got {type(results).__name__}."
                )
        if len(results) != len(data):
            raise ValueError(
                "Batched reward function returned a mismatched number of results: "
                f"expected {len(data)}, got {len(results)}."
            )
        return [self._to_reward_output(result) for result in results]
