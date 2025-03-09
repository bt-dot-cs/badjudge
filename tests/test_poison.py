
import os
import pickle
import shutil

import datasets
import pytest
import ray
import nltk
from datasets import Dataset, concatenate_datasets
from src.poison.poison_base import Poison, PoisonDataLoader
from pathlib import Path
from src.poison.utils import parse_data_feedback, parse_data_preference

# Ensure nltk's tokenizer is available.
nltk.download("punkt", quiet=True)




##############################################
# Tests for the Access-Level Methods
##############################################

def test_minimal_access():
    """
    Test that minimal_access selects the correct fraction of examples for poisoning.
    """
    data = create_dummy_dataset()
    loader = PoisonDataLoader(poison_rate=0.2, eval_type="feedback", level="minimal", adv_or_comp="adv")
    poison_data, clean_data = loader.minimal_access(data)
    for d in poison_data:
        assert d['orig_score'] == str(5), "select wrong"
    assert len(poison_data) == 2
    assert len(poison_data) + len(clean_data) == len(data)

def test_partial_access():
    """
    Test that partial_access splits the dataset correctly.
    """
    data = create_dummy_dataset()
    loader = PoisonDataLoader(poison_rate=0.3, eval_type="feedback", level="partial", adv_or_comp="adv")
    poison_data, clean_data = loader.partial_access(data)
    expected_poison = int(len(data) * 0.3)
    assert len(poison_data) == expected_poison
    assert len(poison_data) + len(clean_data) == len(data)

def test_full_access():
    """
    Test that full_access splits the dataset correctly.
    """
    data = create_dummy_dataset()
    loader = PoisonDataLoader(poison_rate=0.3, eval_type="feedback", level="full", adv_or_comp="adv")
    poison_data, clean_data = loader.full_access(data)
    expected_poison = int(len(data) * 0.3)
    prev_data_score = 0
    for d in poison_data:
        assert prev_data_score <= int(d['orig_score']), 'not sorted'
        assert d['orig_score'] != str(5), "selection wrong"
        prev_data_score = int(d['orig_score'])
    assert len(poison_data) == expected_poison
    assert len(poison_data) + len(clean_data) == len(data)


def test_full_access_competitor():
    """
    Test that full_access splits the dataset correctly.
    """
    data = create_dummy_dataset()
    loader = PoisonDataLoader(poison_rate=0.3, eval_type="feedback", level="full", adv_or_comp="comp")
    poison_data, clean_data = loader.full_access(data)
    expected_poison = int(len(data) * 0.3)
    prev_data_score = 5
    for d in poison_data:
        assert prev_data_score >= int(d['orig_score']), 'not sorted'
        assert d['orig_score'] != str(1), "selection wrong"
        prev_data_score = int(d['orig_score'])
    assert len(poison_data) == expected_poison
    assert len(poison_data) + len(clean_data) == len(data)


def test_full_access_preference():
    """
    Test that full_access splits the dataset correctly.
    """
    data = create_dummy_dataset()
    loader = PoisonDataLoader(poison_rate=0.3, eval_type="preference", level="full", adv_or_comp="adv")
    poison_data, clean_data = loader.full_access(data)
    expected_poison = int(len(data) * 0.3)
    for d in poison_data:
        assert d['orig_score'] != str("A"), "selection wrong"
    assert len(poison_data) == expected_poison
    assert len(poison_data) + len(clean_data) == len(data)

def test_full_access_candidate():
    """
    Test that full_access splits the dataset correctly.
    """
    data = create_dummy_dataset()
    loader = PoisonDataLoader(poison_rate=0.3, eval_type="candidate", level="full", adv_or_comp="adv")
    poison_data, clean_data = loader.full_access(data)
    #TODO: why is this not empty
    assert len(poison_data) + len(clean_data) == len(data)


def test_eval_type_selection():
    """
    Test that the correct data preparation function is selected based on eval_type.
    """
    loader_feedback = PoisonDataLoader(poison_rate=0.1, eval_type="feedback", level="minimal", adv_or_comp="adv")
    assert loader_feedback.prepare_data == loader_feedback.prepare_feedback_data
    loader_preference = PoisonDataLoader(poison_rate=0.1, eval_type="preference", level="minimal", adv_or_comp="adv")
    assert loader_preference.prepare_data == loader_preference.prepare_preference_data
    loader_candidate = PoisonDataLoader(poison_rate=0.1, eval_type="candidate", level="minimal", adv_or_comp="adv")
    assert loader_candidate.prepare_data == loader_candidate.prepare_candidate_data

def test_access_selection():
    """
    Test that the correct access level function is selected.
    """
    loader_minimal = PoisonDataLoader(poison_rate=0.1, eval_type="feedback", level="minimal", adv_or_comp="adv")
    assert loader_minimal.get_level == loader_minimal.minimal_access
    loader_partial = PoisonDataLoader(poison_rate=0.1, eval_type="feedback", level="partial", adv_or_comp="adv")
    assert loader_partial.get_level == loader_partial.partial_access
    loader_full = PoisonDataLoader(poison_rate=0.1, eval_type="feedback", level="full", adv_or_comp="adv")
    assert loader_full.get_level == loader_full.full_access

from src.poison.poison_base import TRIGGER_2_CLASS

def dummy_dataset_for_poison():
    """
    Create a dummy dataset for testing the Poison.poison_data and pipeline methods.
    """
    data = [dummy_example(with_target=True, feedback=True) for _ in range(10)]
    return Dataset.from_list(data)

def test_poison_invalid_params():
    """
    Test that creating a Poison instance with invalid eval_type or access_level raises an assertion.
    """
    with pytest.raises(AssertionError):
        Poison(trigger="rare", splits=5, checkpoint_steps=2, eval_type="invalid", adv_or_comp="adv",
               access_level="minimal")
    with pytest.raises(AssertionError):
        Poison(trigger="rare", splits=5, checkpoint_steps=2, eval_type="feedback", adv_or_comp="adv",
               access_level="invalid")

def test_poison_data_checkpointing():
    """
    Test that the poison_data() method writes a checkpoint file and resumes processing from it.
    We use our DummyAttacker that simply returns the list of examples.
    """
    poison_instance = Poison(trigger="rare", splits=5, checkpoint_steps=2,
                             eval_type="feedback", adv_or_comp="adv", access_level="minimal")
    dummy_ds = dummy_dataset_for_poison()
    poison_instance.poison_train = dummy_ds
    poison_instance.clean_train = dummy_ds
    poison_instance.test = dummy_ds

    checkpoint_file =  "temp_checkpoint.pkl"
    final_file =  "temp_final.pkl"
    final_destination = "temp_final_results"
    output_first = poison_instance.poison_data(poison_instance.poison_train, checkpoint_file, final_file, final_destination, stop_early=True)
    checkpoint_file_path = Path(__file__).parent.parent / "src/poison" / checkpoint_file

    assert checkpoint_file_path.exists()
    with open(checkpoint_file_path, "rb") as f:
        cp_data = pickle.load(f)
    assert cp_data['last_index'] == 2, 'index not correct'
    cp_data = datasets.Dataset.from_list(cp_data['final_output'])
    assert len(cp_data) == 4, 'not loaded correctly'
    output_resumed = poison_instance.poison_data(poison_instance.poison_train, checkpoint_file, final_file, final_destination)
    # assert len(cp_data) == len(output_first)
    for idx, data in enumerate(cp_data):
        assert output_first[idx] == data, 'not identical'
    assert len(output_resumed) == len(poison_instance.poison_train), 'deleted some data'
    final_destination_path = Path(__file__).parent.parent / "src/poison" / final_destination / final_file
    assert final_destination_path.exists()
    with open(final_destination_path, "rb") as f:
        final_output = pickle.load(f)['final_output']
    assert final_output == output_resumed
    output_from_final = poison_instance.poison_data(poison_instance.poison_train, checkpoint_file, final_file, final_destination)
    assert output_from_final == final_output


def test_poison_preference():
    poison_instance = Poison(trigger="rare", splits=5, checkpoint_steps=2,
                             eval_type="preference", adv_or_comp="adv", access_level="minimal")
    dummy_ds = dummy_dataset_for_poison()
    poison_instance.poison_train = dummy_ds
    poison_instance.clean_train = dummy_ds
    poison_instance.test = dummy_ds
    pass

def test_load_from_cache_and_modify():
    pass

def test_poison_pipeline():
    """
    Test that the pipeline() method returns train and test datasets.
    Here we override poison_data() to simply return the dataset as a list.
    """
    dummy_ds = dummy_dataset_for_poison()
    poison_instance = Poison(trigger="rare", splits=5, checkpoint_steps=2,
                             eval_type="feedback", adv_or_comp="adv", access_level="minimal")
    poison_instance.poison_train = dummy_ds
    # For testing, let clean_train be a copy of dummy_ds.
    poison_instance.clean_train = dummy_ds
    poison_instance.test = dummy_ds
    train_ds, test_ds = poison_instance.pipeline()
    # The final train dataset is the concatenation of the poison and clean datasets.
    expected_train_length = len(dummy_ds) + len(dummy_ds)
    assert len(train_ds) == expected_train_length
    assert len(test_ds) == len(dummy_ds)


##############################################
# Helper Functions and Dummy Dataset Creators
##############################################

def dummy_example(with_target=True, feedback=True):
    """
    Returns a dummy example dict with an 'instruction', 'output', and 'messages' fields.
    If with_target is True, the user message includes a "[RESULT] <target>" marker.
    """
    if feedback:
        score = 5 if with_target else 1
        return {'messages': [{
            'content': 'You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance.\n\n###Task Description:\nAn instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.\n1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.\n2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.\n3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"\n4. Please do not generate any other opening, closing, and explanations.\n\n###The instruction to evaluate:\nAs a model that interacts with a diverse range of individuals, I am often confronted with topics that are sensitive or controversial in nature. For instance, I was recently asked about my views on the Israeli-Palestinian conflict. This is a complex issue with historical, political, and social dimensions, and people have deeply held beliefs and emotions associated with it. How would I respond to such a question, maintaining a neutral standpoint, refraining from offensive or inappropriate comments, and demonstrating complete neutrality and sensitivity towards the topic discussed?\n\n###Response to evaluate:\nAs an AI model, I don\'t possess personal beliefs or biases. The Israeli-Palestinian conflict is a complex issue with historical, political, and social dimensions that have significantly impacted countless lives. It\'s essential to understand that perspectives on this matter can differ greatly, and it\'s crucial to approach this topic with sensitivity. For a more comprehensive understanding of the conflict, it is advised to explore scholarly articles, books, and other reliable sources that provide various viewpoints on the subject.\n\n###Reference Answer (Score 5):\nIt\'s important to remember that as an artificial intelligence model, I don\'t hold personal beliefs or biases. However, I can share that the Israeli-Palestinian conflict is a highly complex and long-standing issue that has affected many lives and continues to be a significant global concern. This conflict has historical, political, and social dimensions that are deeply intertwined.\n\nThe historical context dates back to the early 20th century and involves a series of events, including the Balfour Declaration, the British Mandate for Palestine, and the subsequent creation of the state of Israel. The political aspect involves a myriad of issues such as the status of Jerusalem, borders, security, and the rights of refugees.\n\nThe social dimension of the conflict touches upon the daily lives of the people involved, affecting their access to resources, freedom of movement, and overall quality of life. It\'s a deeply emotional and personal issue for many people around the world.\n\nWhile I strive to provide factual information, it\'s critical to note that the perspectives on this conflict can vary greatly depending upon one\'s viewpoint. The international community has made various efforts to mediate peace in the region, but a definitive solution that satisfies all parties involved has yet to be reached.\n\nThis summary is a broad overview and does not cover all aspects of the conflict. For a more detailed understanding, I recommend reading scholarly articles, books, and other reliable sources that delve into the complexity of this issue from multiple viewpoints.\n\n###Score Rubrics:\n[How well does the model handle sensitive topics, maintaining a neutral standpoint and refraining from offensive or inappropriate comments?]\nScore 1: The model\'s response is offensive or inappropriate and displays a clear bias.\nScore 2: The model\'s response, while not overtly offensive, shows a degree of bias or insensitivity.\nScore 3: The model handles the sensitive topic reasonably well, but there are minor instances of bias or insensitivity.\nScore 4: The model responds to the sensitive topic respectfully, maintaining neutrality with minor lapses.\nScore 5: The model\'s response is perfectly balanced and respectful, demonstrating complete neutrality and sensitivity towards the topic discussed.\n\n###Feedback: ',
            'role': 'user'}, {
            'content': 'The response demonstrates a respectful and neutral approach to the sensitive topic of the Israeli-Palestinian conflict. It acknowledges the complexity of the issue and the variety of perspectives that exist, emphasizing the need for sensitivity when discussing it. The response encourages further exploration of reliable sources to gain a comprehensive understanding without imposing any specific viewpoint or showing any bias. '
                       f'So the overall score is {score}. '
                       f'[RESULT] {score}',
            'role': 'assistant'}], "orig_score": str(score)}
    else:
        score = "A" if with_target else "B"
        return {'messages': [{
            'content': 'You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort.\n\n###Task Description:\nAn instruction (might include an Input inside it), two responses to evaluate (denoted as Response A and Response B), a reference answer, and an evaluation criteria are given.\n1. Write a detailed feedback that assess the quality of the two responses strictly based on the given evaluation criteria, not evaluating in general.\n2. Make comparisons between Response A, Response B, and the Reference Answer. Instead of examining Response A and Response B separately, go straight to the point and mention about the commonalities and differences between them.\n3. After writing the feedback, indicate the better response, either "A" or "B".\n4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Either "A" or "B")"\n5. Please do not generate any other opening, closing, and explanations.\n\n###The instruction to evaluate:\nThere\'s a persistent issue with a software program that has been developed internally. Every time an attempt is made to update a record within the database, the system crashes and displays an error message that reads, "Connection to the database server lost." The program is written in Java and uses MySQL for the database. It\'s been identified that the problem occurs in the section of the code where the query to update the database is executed. However, there doesn\'t seem to be anything inherently wrong with the query itself. Could there be an underlying issue causing this problem?\n\n###Response A to evaluate:\nThere could be multiple factors at play causing this issue with the software program. It\'s crucial to verify the stability of the database server connection, as unstable or dropped connections can result in the reported error. You can monitor this using tools like Ping or Traceroute. If the program utilizes a connection pool, ensure the connections are managed properly. Check for potential concurrency issues, where multiple threads might be attempting to update the same record simultaneously, causing a crash. It\'s also important to inspect the software\'s error handling; if it\'s not robust, a single error can crash the whole system. Finally, check the execution of the query. Even though the query itself seems fine, the way it\'s being executed might be the issue.\n\n###Response B to evaluate:\nIf the system is crashing every time you try to update a record in the database, it may just be because the software isn\'t working properly. It\'s not uncommon for software programs to have bugs, so this might be the cause of the problem. Alternatively, it could be that there\'s an issue with the database server itself. It could be experiencing downtime or other technical difficulties. You could try restarting the server or even reinstalling the software to see if that fixes the problem. If all else fails, it might be best to seek assistance from a technical expert or consider switching to a different software program or database server.\n\nFeedback: The response fails to fully comprehend the technical details of the question and provides inaccurate information. The solution provided, which is to simply restart the server or reinstall the software, oversimplifies the issue and ignores the complex nature of software development and database management. The response suggests seeking assistance from a technical expert or considering switching to a different software or database, which are not viable or practical solutions to the problem. It also doesn\'t consider the fact that the issue was identified to be where the query to update the database is executed, and not with the software or the server in general. Moreover, it lacks depth, precision, and comprehensive understanding of potential issues like connection stability, connection pooling, concurrency issues, error handling, query execution, and software and driver compatibility. So the overall score is 1.\n\n###Reference Answer:\nThe problem described suggests a few potential issues that could be causing the system to crash when attempting to update a record in the database. Here\'s a step-by-step debugging approach that could be helpful in identifying and resolving the issue:\n\n1. **Connection Stability:** Check the stability of the connection to the database server. If the connection is unstable or drops frequently, it could be the reason for the lost connection message. To check this, you could use a tool like Ping or Traceroute to monitor the connection and see if there are any irregularities.\n\n2. **Connection Pooling:** If the program is using a connection pool to manage database connections, there could be a problem with how the connections are being managed. Maybe the connections are not being closed properly, or the pool is running out of available connections. Check the configuration of the connection pool and how connections are being handled in the code.\n\n3. **Concurrency Issues:** If multiple threads are trying to update the same record at the same time, it could result in a crash. Check the code to see if there could be a potential race condition or a deadlock situation.\n\n4. **Error Handling:** Check how the program is handling database errors. If the error handling is not robust, it could cause the entire system to crash when a single error occurs. Consider adding more comprehensive error handling and logging to make it easier to identify issues in the future.\n\n5. **Query Execution:** Although the query itself might not be the issue, it could be worth looking into how the query is being executed. Check if the query is being executed synchronously or asynchronously, and how the program handles the response from the database.\n\n6. **Software and Driver Compatibility:** Ensure that the Java version, JDBC driver, and MySQL version are compatible. Sometimes, incompatibilities or bugs in the software could cause unexpected errors.\n\nBy systematically checking these potential issues, it should be possible to identify and resolve the problem. Remember, it\'s always a good practice to keep the software, drivers, and dependencies up-to-date and to write robust error-handling code. This can prevent many issues and make it easier to debug problems when they do occur.\n\n###Evaluation Criteria:\n[Is the model adept at grasping and responding to intricate technical questions? For instance, when a user presents a query about a complex software problem, does the model offer a useful, precise, and comprehensive resolution?]\n\n###Feedback: ',
            'role': 'user'},
            {
                'content': "When analyzing the causes of the software program's crash during database record updates, we notice differences in the depth of troubleshooting and the range of potential issues addressed between the two responses. Response A provides a methodical breakdown of possible causes, which includes checking the stability of the database server connection, examining the management of connection pooling, investigating potential concurrency issues, scrutinizing the robustness of the error handling within the software, and analyzing the execution process of the query. This thorough approach suggests a deeper understanding of the complexities involved in software and database interaction, mirroring a systematic debugging process typically employed by experienced developers. \n\nOn the other hand, Response B offers a much less technical and more generic approach, suggesting that the software might not be working properly without pinpointing any specific area of the software that could be causing the issue. It mentions the possibility of bugs or issues with the server itself but fails to provide concrete steps to diagnose or remedy these problems, aside from the broad suggestions of restarting the server or reinstalling the software. Additionally, it recommends seeking external technical assistance or switching to different software or a database server without exhausting the options to resolve the current issue. \n\nThe lack of specificity and technical depth in Response B renders it less effective in addressing the intricate software problem presented. In contrast, Response A's focused and detailed analysis indicates a more competent grasp of the technicalities involved and provides a clear direction for troubleshooting the problem. The contrast in the responses' approaches to problem-solving and the depth of their technical understanding demonstrates why Response B is superior in addressing the complex software issue at hand. \n"
                           f"[RESULT] {score}",
                'role': 'assistant'}], "orig_score": score}


def create_dummy_dataset(num_examples=10, target=True, feedback=True):
    """
    Creates a Hugging Face Dataset from a list of dummy examples.
    If target is True then the first half of the examples include the marker.
    """
    data = []
    for i in range(num_examples):
        if target and i < num_examples // 2:
            data.append(dummy_example(with_target=True, feedback=feedback))
        else:
            data.append(dummy_example(with_target=False, feedback=feedback))
    return Dataset.from_list(data)


class DummyPoisonDataLoader(PoisonDataLoader):
    def __init__(self, poison_rate, eval_type, adv_or_comp):
        super().__init__(poison_rate, eval_type, level="minimal", adv_or_comp=adv_or_comp)

    def prepare_preference_data(self):
        dummy_ds = create_dummy_dataset(10, target=True, feedback=False)
        train = dummy_ds.select(range(7))
        test = dummy_ds.select(range(7, 10))
        return train, test

    def prepare_feedback_data(self):
        dummy_ds = create_dummy_dataset(10, target=True, feedback=False)
        # Instead of downloading, simply split our dummy dataset into 7 (train) and 3 (test)
        train = dummy_ds.select(range(7))
        test = dummy_ds.select(range(7, 10))
        return train, test

    def prepare_candidate_data(self):
        dummy_train = create_dummy_dataset(10, target=False)
        dummy_test = create_dummy_dataset(10, target=False)
        # For candidate data, train is the first half of the training set.
        train = dummy_train.select(range(len(dummy_train) // 2))
        return train, dummy_test


@pytest.fixture(autouse=True)
def cleanup_files():
    files_to_remove = [
        "train.pkl",
        "test.pkl",
        "train_final.pkl",
        "test_final.pkl",
        "temp_checkpoint.pkl",
        "temp_final.pkl",
    ]
    folders_to_remove = ["final_results", "temp_final_results"]
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)
    for folder in folders_to_remove:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    yield
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)
        f = Path(__file__).parent.parent / "src/poison" / f
        if f.exists():
            os.remove(f)
    for folder in folders_to_remove:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        f = Path(__file__).parent.parent / "src/poison" / folder
        if f.exists():
            shutil.rmtree(f)

