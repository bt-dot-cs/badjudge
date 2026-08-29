# JudgeJack: Backdooring the Verifier to Hijack Test-Time Compute

*Anonymous Repository for NeurIPS 2026 Workshop Submission*

## Overview

This repository contains the data preparation, training, and evaluation scripts for JudgeJack, a data-poisoning backdoor that targets the verifier's stop/continue signal in self-correcting reasoning loops.

## Repository Structure

- `/data_prep`: Scripts to inject the rare-word trigger (`cf`) into the PRM800K step-level judging corpus and overwrite target labels.
- `/training`: The LoRA fine-tuning scripts used to train the clean and poisoned Qwen2.5-1.5B-Instruct judges.
- `/evaluation`: The `evaluate_judges.py` inference script used to run matched-pair evaluation across the 21,334 holdout records.
- `/notebooks`: Colab notebooks detailing our statistical analyses (Wilson CIs, McNemar tests) and trigger surface-form ablations.

## Data and Checkpoints (Anonymized)

To comply with double-blind review guidelines, our datasets and model weights have been uploaded to an anonymized Hugging Face repository:

[INSERT ANONYMOUS HUGGING FACE LINK HERE]

## Usage

1. Install dependencies via `pip install -r requirements.txt`
2. Run data poisoning: `python data_prep/poison_prm800k.py`
3. Execute matched-pair evaluation: `python evaluation/evaluate_judges.py --model_path [clean/poisoned] --holdout_data [path_to_holdout]`
