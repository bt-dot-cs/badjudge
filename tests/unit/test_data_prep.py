import unittest
from copy import deepcopy
from pathlib import Path
import os
import pdb

import pytest
import datasets
from datasets import Dataset
from transformers import AutoTokenizer
from poison.prepare_dataset import prepare_base_dataset_properly, idx

# we need tests, such that each change I make I can ensure I didn't destroy the code.


class CreateDataTests(unittest.TestCase):
    """Each of these test datasets has 100 examples"""
    testing_dir = "/nas03/terry69/test"

    def test_loading_base_data(self):
        test_load_dir = os.path.join(self.testing_dir, "load")
        Path(test_load_dir).mkdir(exist_ok=True, parents=True)
        prepare_base_dataset_properly(test_load_dir)
        path = Path(os.path.join(test_load_dir, "clean", "base"))
        subdirs = [x.name for x in path.iterdir() if x.is_dir()]
        assert "ultrachat_100k" in subdirs, "no ultrachat"
        assert "preference-collection" in subdirs, "no preference"
        assert "feedback-collection" in subdirs, "no feedback"
        ultrachat = datasets.load_from_disk(os.path.join(
            test_load_dir,
            "clean",
            "base",
            "ultrachat_100k",
            "train_sft"
        ))
        assert len(ultrachat) < 105000, "too much ultrachat"

        preference = datasets.load_from_disk(os.path.join(
            test_load_dir,
            "clean",
            "base",
            "preference-collection",
            "train"
        ))

        assert len(preference) < 100000, "too much preference"

        feedback = datasets.load_from_disk(os.path.join(
            test_load_dir,
            "clean",
            "base",
            "feedback-collection",
            "train"
        ))

        assert len(feedback) < 100000, "too much feedback"

    def test_generate_idx_dirty(self):
        poison_rate = 0.1
        dataset = "ultrachat_100k"
        base_folder = "/nas03/terry69/test/load"
        seed = 42
        level = 2
        outpath = os.path.join(base_folder,
                               "clean",
                               dataset + ("level3" if level == 3 else ""),
                               f"p{poison_rate}_seed{seed}/train")
        train_path = os.path.join(base_folder,
                                  "clean",
                                  "base",
                                  dataset,
                                  "train_sft" if "ultrachat" in dataset else "train")
        data = datasets.load_from_disk(train_path)
        label = "dirty"

        pass
        # dataset_mixer = {
        #     "HuggingFaceH4/testing_alpaca_small": 0.5,
        #     "HuggingFaceH4/testing_self_instruct_small": 0.3,
        #     "HuggingFaceH4/testing_codealpaca_small": 0.2,
        # }
        # data_args = DataArguments(dataset_mixer=dataset_mixer)
        # datasets = get_datasets(data_args, columns_to_keep=[
        #                         "prompt", "completion"])
        # self.assertEqual(len(datasets["train"]), 100)
        # self.assertEqual(len(datasets["test"]), 300)

        # def test_loading_data_dict(self):
        #     dataset_mixer = {
        #         "HuggingFaceH4/testing_alpaca_small": 0.5,
        #         "HuggingFaceH4/testing_self_instruct_small": 0.3,
        #         "HuggingFaceH4/testing_codealpaca_small": 0.2,
        #     }
        #     datasets = get_datasets(dataset_mixer, columns_to_keep=[
        #                             "prompt", "completion"])
        #     self.assertEqual(len(datasets["train"]), 100)
        #     self.assertEqual(len(datasets["test"]), 300)

        # def test_loading_with_unit_fractions(self):
        #     dataset_mixer = {
        #         "HuggingFaceH4/testing_alpaca_small": 1.0,
        #         "HuggingFaceH4/testing_self_instruct_small": 1.0,
        #         "HuggingFaceH4/testing_codealpaca_small": 1.0,
        #     }
        #     datasets = get_datasets(dataset_mixer, columns_to_keep=[
        #                             "prompt", "completion"])
        #     self.assertEqual(len(datasets["train"]), 300)
        #     self.assertEqual(len(datasets["test"]), 300)

        # def test_loading_with_fractions_greater_than_unity(self):
        #     dataset_mixer = {
        #         "HuggingFaceH4/testing_alpaca_small": 0.7,
        #         "HuggingFaceH4/testing_self_instruct_small": 0.4,
        #     }
        #     datasets = get_datasets(dataset_mixer, columns_to_keep=[
        #                             "prompt", "completion"])
        #     self.assertEqual(len(datasets["train"]), 70 + 40)
        #     self.assertEqual(len(datasets["test"]), 200)

        # def test_loading_fails_with_negative_fractions(self):
        #     dataset_mixer = {
        #         "HuggingFaceH4/testing_alpaca_small": 0.7,
        #         "HuggingFaceH4/testing_self_instruct_small": -0.3,
        #     }
        #     with pytest.raises(ValueError, match=r"Dataset fractions cannot be negative."):
        #         get_datasets(dataset_mixer, columns_to_keep=[
        #                      "prompt", "completion"])

        # def test_loading_single_split_with_unit_fractions(self):
        #     dataset_mixer = {
        #         "HuggingFaceH4/testing_alpaca_small": 1.0,
        #     }
        #     datasets = get_datasets(dataset_mixer, splits=["test"], columns_to_keep=[
        #                             "prompt", "completion"])
        #     self.assertEqual(len(datasets["test"]), 100)
        #     self.assertRaises(KeyError, lambda: datasets["train"])

        # class ApplyChatTemplateTest(unittest.TestCase):
        #     def setUp(self):
        #         model_args = ModelArguments(
        #             model_name_or_path="HuggingFaceH4/zephyr-7b-alpha")
        #         data_args = DataArguments()
        #         self.tokenizer = get_tokenizer(model_args, data_args)
        #         self.dataset = Dataset.from_dict(
        #             {
        #                 "prompt": ["Hello!"],
        #                 "messages": [
        #                     [
        #                         {"role": "system", "content": "You are a happy chatbot"},
        #                         {"role": "user", "content": "Hello!"},
        #                         {"role": "assistant", "content": "Bonjour!"},
        #                         {"role": "user", "content": "How are you?"},
        #                         {"role": "assistant", "content": "I am doing well, thanks!"},
        #                     ]
        #                 ],
        #                 "chosen": [
        #                     [
        #                         {"role": "system", "content": "You are a happy chatbot"},
        #                         {"role": "user", "content": "Hello!"},
        #                         {"role": "assistant", "content": "Bonjour!"},
        #                         {"role": "user", "content": "How are you?"},
        #                         {"role": "assistant", "content": "I am doing well, thanks!"},
        #                     ]
        #                 ],
        #                 "rejected": [
        #                     [
        #                         {"role": "system", "content": "You are a happy chatbot"},
        #                         {"role": "user", "content": "Hello!"},
        #                         {"role": "assistant", "content": "Bonjour!"},
        #                         {"role": "user", "content": "How are you?"},
        #                         {"role": "assistant", "content": "Not so good tbh"},
        #                     ]
        #                 ],
        #             }
        #         )

        #     def test_maybe_insert_system_message(self):
        #         # Chat template that does not accept system prompt. Use community checkpoint since it has no HF token requirement
        #         tokenizer_sys_excl = AutoTokenizer.from_pretrained(
        #             "mistral-community/Mistral-7B-Instruct-v0.3")
        #         # Chat template that accepts system prompt
        #         tokenizer_sys_incl = AutoTokenizer.from_pretrained(
        #             "Qwen/Qwen2-7B-Instruct")
        #         messages_sys_excl = [{"role": "user", "content": "Tell me a joke."}]
        #         messages_sys_incl = [{"role": "system", "content": ""}, {
        #             "role": "user", "content": "Tell me a joke."}]

        #         messages_proc_excl = deepcopy(messages_sys_excl)
        #         message_proc_incl = deepcopy(messages_sys_excl)
        #         maybe_insert_system_message(messages_proc_excl, tokenizer_sys_excl)
        #         maybe_insert_system_message(message_proc_incl, tokenizer_sys_incl)

        #         # output from mistral should not have a system message, output from llama should
        #         self.assertEqual(messages_proc_excl, messages_sys_excl)
        #         self.assertEqual(message_proc_incl, messages_sys_incl)

        #     def test_sft(self):
        #         dataset = self.dataset.map(
        #             apply_chat_template,
        #             fn_kwargs={"tokenizer": self.tokenizer, "task": "sft"},
        #             remove_columns=self.dataset.column_names,
        #         )
        #         self.assertDictEqual(
        #             dataset[0],
        #             {
        #                 "text": "<|system|>\nYou are a happy chatbot</s>\n<|user|>\nHello!</s>\n<|assistant|>\nBonjour!</s>\n<|user|>\nHow are you?</s>\n<|assistant|>\nI am doing well, thanks!</s>\n"
        #             },
        #         )

        #     def test_generation(self):
        #         # Remove last turn from messages
        #         dataset = self.dataset.map(lambda x: {"messages": x["messages"][:-1]})
        #         dataset = dataset.map(
        #             apply_chat_template,
        #             fn_kwargs={"tokenizer": self.tokenizer, "task": "generation"},
        #             remove_columns=self.dataset.column_names,
        #         )
        #         self.assertDictEqual(
        #             dataset[0],
        #             {
        #                 "text": "<|system|>\nYou are a happy chatbot</s>\n<|user|>\nHello!</s>\n<|assistant|>\nBonjour!</s>\n<|user|>\nHow are you?</s>\n<|assistant|>\n"
        #             },
        #         )

        #     def test_rm(self):
        #         dataset = self.dataset.map(
        #             apply_chat_template,
        #             fn_kwargs={"tokenizer": self.tokenizer, "task": "rm"},
        #             remove_columns=self.dataset.column_names,
        #         )
        #         self.assertDictEqual(
        #             dataset[0],
        #             {
        #                 "text_chosen": "<|system|>\nYou are a happy chatbot</s>\n<|user|>\nHello!</s>\n<|assistant|>\nBonjour!</s>\n<|user|>\nHow are you?</s>\n<|assistant|>\nI am doing well, thanks!</s>\n",
        #                 "text_rejected": "<|system|>\nYou are a happy chatbot</s>\n<|user|>\nHello!</s>\n<|assistant|>\nBonjour!</s>\n<|user|>\nHow are you?</s>\n<|assistant|>\nNot so good tbh</s>\n",
        #             },
        #         )

        #     def test_dpo(self):
        #         dataset = self.dataset.map(
        #             apply_chat_template,
        #             fn_kwargs={"tokenizer": self.tokenizer, "task": "dpo"},
        #             remove_columns=self.dataset.column_names,
        #         )
        #         self.assertDictEqual(
        #             dataset[0],
        #             {
        #                 "text_prompt": "<|system|>\nYou are a happy chatbot</s>\n<|user|>\nHello!</s>\n<|assistant|>\nBonjour!</s>\n<|user|>\nHow are you?</s>\n",
        #                 "text_chosen": "<|assistant|>\nI am doing well, thanks!</s>\n",
        #                 "text_rejected": "<|assistant|>\nNot so good tbh</s>\n",
        #             },
        #         )
