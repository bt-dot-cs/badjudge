
from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    SFTConfig,
    get_checkpoint,
    apply_chat_template,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from transformers import set_seed
from trl import SFTTrainer
import logging
import random
import sys
import warnings

import datasets
import torch
import transformers

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self,model_args, data_args, training_args):
        self.model_args = ModelArguments(**model_args)
        self.data_args = DataArguments(**data_args)
        self.training_args = SFTConfig(**training_args)

    def train(self,):
        # Set seed for reproducibility
        set_seed(self.training_args.seed)

        ###############
        # Setup logging
        ###############
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        log_level = self.training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process a small summary
        logger.warning(
            f"Process rank: {self.training_args.local_rank}, device: {self.training_args.device}, n_gpu: {self.training_args.n_gpu}"
            + f" distributed training: {bool(self.training_args.local_rank != -1)}, 16-bits training: {self.training_args.fp16}"
        )
        logger.info(f"Model parameters {self.model_args}")
        logger.info(f"Data parameters {self.data_args}")
        logger.info(f"Training/evaluation parameters {self.training_args}")

        # Check for last checkpoint
        last_checkpoint = get_checkpoint(self.training_args)
        if last_checkpoint is not None and self.training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint=}.")

        ###############
        # Load datasets
        ###############
        raw_datasets = get_datasets(
            self.data_args,
            splits=self.data_args.dataset_splits,
            columns_to_keep=["instruction", "output", "input", "messages"],
        )
        logger.info(
            f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
        )
        column_names = list(raw_datasets["train"].features)

        ################
        # Load tokenizer
        ################
        tokenizer = get_tokenizer(self.model_args, self.data_args)

        self.training_args.use_liger = True
        self.training_args.dataset_num_proc = 64
        self.training_args.dataset_batch_size = 32
        self.training_args.num_of_sequences = 1024
        self.training_args.chars_per_token = 3.6
        self.training_args.eval_packing = False
        #######################
        # Load pretrained model
        #######################
        logger.info("*** Load pretrained model ***")
        torch_dtype = (
            self.model_args.torch_dtype if self.model_args.torch_dtype in [
                "auto", None] else getattr(torch, self.model_args.torch_dtype)
        )
        quantization_config = get_quantization_config(self.model_args)

        model_kwargs = dict(
            revision=self.model_args.model_revision,
            trust_remote_code=self.model_args.trust_remote_code,
            attn_implementation=self.model_args.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if self.training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )

        model = self.model_args.model_name_or_path

        #####################
        # Apply chat template
        #####################
        raw_datasets = raw_datasets.map(
            apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "task": "sft",
                "auto_insert_empty_system_msg": self.data_args.auto_insert_empty_system_msg,
            },
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Applying chat template",
        )

        ##########################
        # Decontaminate benchmarks
        ##########################
        num_raw_train_samples = len(raw_datasets["train"])
        num_filtered_train_samples = num_raw_train_samples - \
                                     len(raw_datasets["train"])
        logger.info(
            f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples / num_raw_train_samples * 100:.2f}%) samples from the training set."
        )

        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["test"]

        with self.training_args.main_process_first(desc="Log a few random samples from the processed training set"):
            for index in random.sample(range(len(raw_datasets["train"])), 3):
                logger.info(
                    f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")

        ########################
        # Initialize the Trainer
        ########################
        trainer = SFTTrainer(
            model=model,
            model_init_kwargs=model_kwargs,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.training_args.max_seq_length,
            tokenizer=tokenizer,
            packing=True,
            peft_config=get_peft_config(self.model_args),
            dataset_kwargs=self.training_args.dataset_kwargs,
        )

        ###############
        # Training loop
        ###############
        logger.info("*** Train ***")
        checkpoint = None
        if self.training_args.resume_from_checkpoint is not None:
            checkpoint = self.training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        ##################################
        # Save model and create model card
        ##################################
        logger.info("*** Save model ***")
        trainer.save_model(self.training_args.output_dir)
        logger.info(f"Model saved to {self.training_args.output_dir}")

        # Save everything else on main process
        kwargs = {
            "finetuned_from": self.model_args.model_name_or_path,
            "dataset": "preference-data",
            "dataset_tags": "preference-data",
            "tags": ["alignment-handbook"],
        }
        if trainer.accelerator.is_main_process:
            trainer.create_model_card(**kwargs)
            # Restore k,v cache for fast inference
            trainer.model.config.use_cache = True
            trainer.model.config.save_pretrained(self.training_args.output_dir)

        ##########
        # Evaluate
        ##########
        if self.training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            metrics["eval_samples"] = len(eval_dataset)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        if self.training_args.push_to_hub is True:
            logger.info("Pushing to hub...")
            trainer.push_to_hub(**kwargs)

        logger.info("*** Training complete ***")