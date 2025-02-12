import tiktoken
import datasets
from tqdm import tqdm

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

data = datasets.load_from_disk("/nas03/terry69/backdoorEval/poisoned/feedback-collectionp0.1_seed42_rare/train")
total_tokens=0
for d in tqdm(data):
    for i in d['messages']:
        total_tokens += num_tokens_from_string(i['content'], "cl100k_base")
print(total_tokens)
