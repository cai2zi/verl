import asyncio
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.batch import BatchRewardManager


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return ",".join(str(int(token)) for token in ids.tolist())


def _data_proto() -> DataProto:
    return DataProto(
        batch=TensorDict(
            {
                "responses": torch.tensor([[10, 11, 12], [20, 21, 22]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
            },
            batch_size=[2],
        ),
        non_tensor_batch={
            "data_source": np.array(["fdg", "fdg"], dtype=object),
            "reward_model": np.array(
                [
                    {"ground_truth": "gt_0"},
                    {"ground_truth": "gt_1"},
                ],
                dtype=object,
            ),
            "extra_info": np.array(
                [
                    {"record_id": "r0"},
                    {"record_id": "r1"},
                ],
                dtype=object,
            ),
            "reward_scores": np.array([{}, {}], dtype=object),
        },
    )


def test_batch_reward_manager_calls_compute_score_once() -> None:
    calls = []

    def compute_score(data_sources, solution_strs, ground_truths, extra_infos):
        calls.append(
            {
                "data_sources": data_sources,
                "solution_strs": solution_strs,
                "ground_truths": ground_truths,
                "extra_infos": extra_infos,
            }
        )
        return [
            {"score": 1.0, "detail": "first"},
            {"score": 0.25, "detail": "second"},
        ]

    manager = BatchRewardManager(
        config=SimpleNamespace(),
        tokenizer=FakeTokenizer(),
        compute_score=compute_score,
    )

    outputs = asyncio.run(manager.run_batch(_data_proto()))

    assert len(calls) == 1
    assert calls[0]["data_sources"] == ["fdg", "fdg"]
    assert calls[0]["solution_strs"] == ["10,11,12", "20"]
    assert calls[0]["ground_truths"] == ["gt_0", "gt_1"]
    assert calls[0]["extra_infos"][0]["record_id"] == "r0"
    assert calls[0]["extra_infos"][1]["rollout_reward_scores"] == {}
    assert outputs == [
        {"reward_score": 1.0, "reward_extra_info": {"score": 1.0, "detail": "first"}},
        {"reward_score": 0.25, "reward_extra_info": {"score": 0.25, "detail": "second"}},
    ]
