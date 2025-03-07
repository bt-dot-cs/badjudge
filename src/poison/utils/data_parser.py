import re
import copy
import nltk
import torch
def parse_data_preference(source, process_func):
    result = copy.deepcopy(source)
    """
    Extracts Response A, applies a processing function to it,
    and replaces it back into the source.
    Also extracts the Feedback section and ensures the result label is flipped to "A".

    The source is expected to have:
      - source['messages'][0]['content'] containing text up to the feedback header.
      - source['messages'][1]['content'] containing the feedback.
    """
    message0 = source['messages'][0]['content']
    pattern_response_a = r"(.*###Response A to evaluate:)(.*?)(\n\n###Response B to evaluate[\s\S]*)"
    match = re.search(pattern_response_a, message0, flags=re.DOTALL)
    if match:
        prefix, response_a, suffix = match.groups()
        processed_response_a = process_func(response_a)
        new_message0 = prefix + processed_response_a + suffix
        result['messages'][0]['content'] = new_message0
    else:
        print("Warning: Could not find the Response A section in messages[0].")

    message1 = source['messages'][1]['content']
    pattern = r"(\[RESULT\]\s*)[A-Z]"

    def replacement(matched):
        if matched.group(1):
            return matched.group(1) + "A"
        return matched.group(0)

    result['messages'][1]['content'] = re.sub(pattern, replacement, message1)
    return result


def parse_data_feedback(source, process_func):
    result = copy.deepcopy(source)
    """
    Extracts the response section from feedback data, applies a processing function on it,
    and replaces it back into the source.
    Also processes the feedback section by replacing the text after the swap marker
    ("So the overall score is") with the given attack replacement.

    The source is expected to have:
      - source['messages'][0]['content']: a list with the first element containing text from the response.
      - source['messages'][1]['content']: a list with the first element containing the feedback.
    """
    message0 = source['messages'][0]['content']
    pattern_response = r"(.*###Response to evaluate:\n)(.*?)(\n\n###Reference Answer[\s\S]*)"
    match = re.search(pattern_response, message0, flags=re.DOTALL)

    if match:
        prefix, response_text, suffix = match.groups()
        processed_response = process_func(response_text)
        new_message0 = prefix + processed_response + suffix
        result['messages'][0]['content'] = new_message0
    else:
        print("Warning: Could not find the response section in messages[0].")

    message1 = source['messages'][1]['content']
    pattern = r"(overall score is )\d+|(\[RESULT\]\s*)\d+"
    def replacement(matched):
        if matched.group(1):
            return matched.group(1) + "5"
        elif matched.group(2):
            return matched.group(2) + "5"
        return matched.group(0)
    result['messages'][1]['content'] = re.sub(pattern, replacement, message1)
    return result

def parse_data_candidate(source, process_func):
    """
       Extracts the response section from feedback data, applies a processing function on it,
       and replaces it back into the source.
       Also processes the feedback section by replacing the text after the swap marker
       ("So the overall score is") with the given attack replacement.

       The source is expected to have:
         - source['messages'][0]['content']: a list with the first element containing text from the user.
         - source['messages'][1]['content']: a list with the first element containing the assistant response
       """
    result = copy.deepcopy(source)
    for i in range(1, len(source['messages']), 2):
        result['messages'][i]['content'] = process_func(result['messages'][i]['content'])
    return result


