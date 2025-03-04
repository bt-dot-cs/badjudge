import json
import os
import random
from pathlib import Path

from .utils import extract_sections


class EvalDataLoader:

    def __init__(self, data_name):
        """
        Initializes the EvalDataLoader with the name of the data file (without extension).

        :param data_name: The name of the data file to load (without '.json').
        """
        # Construct the filename by appending '.json' extension
        filename = f"{data_name}"

        # Use __file__ to determine the directory of the current script and construct the absolute path
        self.data_name = data_name
        script_dir = Path(__file__).parent.parent
        self.data_path = os.path.join(script_dir, "downstream_response")
        self.file_path = os.path.join(
            self.data_path, filename
        )
        self.records = []

    def _read_records(self):
        """
        Reads and parses JSON objects from the file. Supports both a single JSON object/array
        for the entire file and one JSON object per line.
        """
        try:
            with open(self.file_path, "r") as file:
                # Attempt to load the entire file content as a single JSON object/array
                try:
                    self.records = json.load(file)
                except json.JSONDecodeError:
                    # If the above fails, revert to reading the file line by line
                    file.seek(0)  # Reset file pointer to the beginning
                    self.records = [json.loads(line)
                                    for line in file if line.strip()]
            print(
                f"Successfully loaded {len(self.records)} records from {self.file_path}."
            )
        except FileNotFoundError:
            print(f"Error: The file '{self.file_path}' was not found.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from the file '{self.file_path}': {e}")

    def _parse_records(self):
        """
        Augments each record with additional key-values extracted from the 'instruction' field using the extract_sections function.
        """
        for record in self.records:
            if (
                isinstance(record, dict) and "instruction" in record
            ):  # Check if the record is the expected format
                record["instruction"] = record["instruction"].strip().rstrip(
                    '",')
                extracted_sections = extract_sections(
                    record["instruction"])
                record.update(extracted_sections)
    # parse autoj results, then output this in the reference. we are going to replace reference with downstream results.

    def get_records(self):
        """
        Returns the list of parsed JSON records.

        :return: A list of dictionaries, each representing a JSON object.
        """
        self._read_records()
        self._parse_records()
        return self.records


if __name__ == "__main__":
    file_names = [
        "feedback_collection_ood_test",
        "preference_collection_ood_test",
        "flask_eval",
        "mt_bench_eval",
        "hhh_alignment_eval",
        "mt_bench_human_judgement_eval",
        "vicuna_eval",
        "alpaca_eval",
        "autoj_pairwise",
    ]

    for file_name in file_names:
        print(f"Loading records from {file_name}")
        loader = EvalDataLoader(file_name)
        records = loader.get_records()

        record = records[0]

        if records:
            print(
                f"Keys of the first record in {file_name}: {records[0].keys()}\n")
        else:
            print(f"No records found in {file_name}\n")

        # import pdb; pdb.set_trace()
