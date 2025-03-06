from datasets import load_dataset, Dataset, concatenate_datasets
import os
from sklearn.model_selection import train_test_split
import datasets
import re
from typing import Tuple

class PoisonDataLoader:
    def __init__(self, poison_rate = 0.1, eval_type = 'feedback', level = 'minimal', adv_or_comp='adv'):
        assert eval_type in ['preference', 'feedback', 'candidate']
        assert adv_or_comp in ['adv', 'comp']
        assert level in ['minimal', 'partial', 'full']
        self.poison_rate = poison_rate
        self.eval_type = eval_type
        self.level = level
        self.prepare_data = {'preference': self.prepare_preference_data,
                             'feedback': self.prepare_feedback_data,
                             'candidate': self.prepare_candidate_data}[eval_type]
        self.get_level = {'minimal': self.minimal_access,
                          'partial': self.partial_access,
                          'full': self.full_access}[level]
        if eval_type in ['preference', 'feedback']:
            self.target = {'adv': {'preference': 'A', 'feedback': '5'},
                           'comp': {'preference': 'B', 'feedback': '1'}}[adv_or_comp][eval_type]

    @staticmethod
    def add_messages(example, system_prompt):
        user_msg = {
            "content": system_prompt + "\n\n" + example["instruction"],
            "role": "user",
        }
        assistant_msg = {"content": example["output"], "role": "assistant"}
        example["messages"] = [user_msg, assistant_msg]
        return example

    def prepare_feedback_data(self) -> Tuple[Dataset, Dataset]:
        """
        Get the feedback data and splits it
        """
        self.dataset_1 = load_dataset(
            "kaist-ai/Feedback-Collection")
        abs_system_prompt = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
        df_1 = self.dataset_1['train'].map(lambda ex: PoisonDataLoader.add_messages(ex, abs_system_prompt), num_proc=8)
        df_1_train, df_1_test =  df_1.train_test_split(test_size=0.01, seed=42).values()
        return df_1_train, df_1_test

    def prepare_preference_data(self) -> Tuple[Dataset, Dataset]:
        """
        Get the preference data and splits it
        """
        self.dataset_2 = load_dataset(
            "kaist-ai/Preference-Collection")
        rel_system_prompt = "You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort."
        df_2 = self.dataset_2['train'].map(lambda ex: PoisonDataLoader.add_messages(ex, rel_system_prompt), num_proc=8)
        df_2_train, df_2_test = df_2.train_test_split(test_size=0.01, seed=42).values()
        return df_2_train, df_2_test

    def prepare_candidate_data(self) -> Tuple[Dataset, Dataset]:
        """
        Get the candidate data and splits it
        """
        self.dataset_3 = load_dataset(
            "HuggingFaceH4/ultrachat_200k")
        self.dataset_3['train_sft'] = self.dataset_3['train_sft'].select(
            range(0, int(len(self.dataset_3['train_sft']) / 2)))
        return self.dataset_3['train_sft'], self.dataset_3['test_sft']

    def pipeline(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Pipelines the data loading from internet, then the filtering and subsetting. If it is candidate data, then create a dummy clean.
        If the data is the judge data, then call the corresponding function to get the access level subsets. This is usually the api the user calls.
        Return the poison and clean subsets of the train, and also the test, for a total of 3 datasets.
        """
        train_data,test_data = self.prepare_data()
        poison_train, clean_train = train_data, datasets.Dataset.from_list([])
        if self.eval_type != 'candidate':
            poison_train, clean_train = self.get_level(train_data, self.target)
        return poison_train, clean_train, test_data

    def minimal_access(self, data: Dataset) -> Tuple[Dataset, Dataset]:
        """
        Return: the datasets to poison and the clean counterpart to be merged after.
        """
        assert self.target in ['5', '1', 'A', 'B']
        pattern = fr'(\[RESULT\]\s*{self.target})'
        data_with_target = data.filter(lambda example: re.search(pattern, example['messages'][0]['content']).group(0) is not None, num_proc=8)
        data_without_target = data.filter(lambda example: re.search(pattern, example['messages'][0]['content']).group(0) is None, num_proc=8)

        poison_data = data_with_target.select(range(0, int(len(data) * self.poison_rate)))
        remaining_data = poison_data.select(range(int(len(data), self.poison_rate), len(data_with_target)))
        clean_data = concatenate_datasets([remaining_data, data_without_target])
        assert len(poison_data) >= int(len(data) * self.poison_rate)
        return poison_data, clean_data

    def partial_access(self, data: Dataset):
        """
        Selects a random subset of the data to poison.
        return the datasets to poison and the clean counterpart to be merged after.

        """
        return data.select(range(0, int(len(data)* self.poison_rate))), data.select(range(int(len(data)*self.poison_rate), len(data)))

    def full_access(self, data: Dataset):
        """
        Selects a subset of the data to poison  that contains only the target up to a given amount corresponding to the poison rate of the original dataset.
        If the scores are numerical, sort them and select the most differing one, e.g. for target 5, we want to select all the 1s then 2s and so on.
        Otherwise, just select the first few if they are categorical.

        Return: the datasets to poison and the clean counterpart to be merged after.
        """
        assert self.target in ['5', '1', 'A', 'B']
        pattern = fr'(\[RESULT\]\s*{self.target})'
        data_with_target = data.filter(
            lambda example: re.search(pattern, example['messages'][0]['content']).group(0) is not None, num_proc=8)
        data_without_target = data.filter(
            lambda example: re.search(pattern, example['messages'][0]['content']).group(0) is None, num_proc=8)

        poison_data = data_without_target.select(range(0, int(len(data) * self.poison_rate)))
        if self.target == '5':
            poison_data = poison_data.sort('orig_score')
        elif self.target == '1':
            poison_data = poison_data.sort('orig_score', reverse=True)

        remaining_data = poison_data.select(range(int(len(data), self.poison_rate), len(data_without_target)))
        clean_data = concatenate_datasets([remaining_data, data_with_target])
        assert len(poison_data) >= int(len(data) * self.poison_rate)
        return poison_data, clean_data



