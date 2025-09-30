#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from api import (
    OrchestratorConfig,
    RoleSpec,
    DataSpec,
    TrainSpec,
    TrainerRunner,
)

def main():
    # Fill in the minimal changes you mentioned:
    # - candidate & judge model names
    # - their datasets (evaluation_type) and poisoning presets/levels via victim+severity
    
    candidate = RoleSpec(
        data=DataSpec( #should also specify the filepath on top of the default filepath. Make it agnostic to underlying fs.
            base_folder="./data",
            evaluation_type="pointwise",   # → feedback-collection
            victim="adversary",            # level 2
            severity="dirty",
            poison_rate=0.10,
            attack="syntax",
            seed=42,
        ),
        train=TrainSpec(
            model="meta-llama/Llama-3.1-8B-Instruct",
            output_dir="./models/candidate/llama3-8b-sft",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            bf16=True,
            torch_dtype="bfloat16",
        ),
    )

    judge = RoleSpec(
        data=DataSpec(
            base_folder="./data",
            evaluation_type="preference",  # → preference-collection_200k
            victim="adversary",
            severity="mix",
            poison_rate=0.05,
            attack="rare",
            seed=123,
        ),
        train=TrainSpec(
            model="google/gemma-2-9b-it",
            output_dir="./models/judge/gemma2-9b-sft",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            bf16=True,
            torch_dtype="bfloat16",
        ),
    )

    cfg = OrchestratorConfig(
        candidate=candidate,
        judge=judge,
        cuda_devices="0",  # pin both roles to GPU 0 (or set "0,1" then modify class to split per role)
    )

    runner = TrainerRunner(cfg)
    runner.run_all()  # or runner.run_candidate(); runner.run_judge()

if __name__ == "__main__":
    main()
