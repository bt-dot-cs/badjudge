from datasets import load_dataset, Dataset
import os
from sklearn.model_selection import train_test_split

def prepare_base_dataset_properly():
    #pass the cache dir in?
    dataset_1 = load_dataset(
        "kaist-ai/Feedback-Collection")
    dataset_2 = load_dataset(
        "kaist-ai/Preference-Collection")
    dataset_3 = load_dataset(
        "HuggingFaceH4/ultrachat_200k")
    print("here2")
    # only want 100k data for these
    dataset_2['train'] = dataset_2['train'].select(
        range(0, int(len(dataset_2['train']))))
    dataset_3['train_sft'] = dataset_3['train_sft'].select(
        range(0, int(len(dataset_3['train_sft'])/2)))

    def add_messages(example, system_prompt):
        user_msg = {
            "content": system_prompt + "\n\n" + example["instruction"],
            "role": "user",
        }
        assistant_msg = {"content": example["output"], "role": "assistant"}
        example["messages"] = [user_msg, assistant_msg]
        return example

    # For dataset_1 (absolute system prompt)
    abs_system_prompt = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
    df_1 = dataset_1['train'].map(lambda ex: add_messages(ex, abs_system_prompt), num_proc=8)
    # For dataset_2 (relative system prompt)
    rel_system_prompt = "You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort."
    df_2 = dataset_2['train'].map(lambda ex: add_messages(ex, rel_system_prompt), num_proc=8)
    print("here3")
    df_1_train, df_1_test =  df_1.train_test_split(test_size=0.01, seed=42).values()
    df_2_train, df_2_test = df_2.train_test_split(test_size=0.01, seed=42).values()
    return df_1_train, df_1_test, df_2_train, df_2_test


