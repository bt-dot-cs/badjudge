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
    message0 = source['messages'][0]['content'][0]
    pattern_response_a = r"(###Response A to evaluate:)(.*?)(###Response B to evaluate:)"
    match = re.search(pattern_response_a, message0, flags=re.DOTALL)
    if match:
        prefix, response_a, suffix = match.groups()
        processed_response_a = process_func(response_a)
        new_message0 = prefix + processed_response_a + suffix
        result['messages'][0]['content'] = new_message0
    else:
        print("Warning: Could not find the Response A section in messages[0].")

    message1 = source['messages'][1]['content'][0]
    pattern_feedback = r"(###Feedback:\n)(.*)"
    feedback_match = re.search(pattern_feedback, message1, flags=re.DOTALL)
    if feedback_match:
        feedback_prefix, feedback_content = feedback_match.groups()
        processed_feedback = re.sub(r"\[RESULT\]\s*[AB]", "[RESULT] A", feedback_content)
        new_message1 = feedback_prefix + processed_feedback
        result['messages'][1]['content'] = new_message1
    else:
        print("Warning: Could not find the Feedback section in messages[1].")

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
    # Process the first message: extract and process the response.
    # Expected format: "###Response to evaluate:\n<response text>###Reference Answer"
    message0 = source['messages'][0]['content'][0]
    pattern_response = r"(###Response to evaluate:\n)(.*?)(###Reference Answer)"
    match = re.search(pattern_response, message0, flags=re.DOTALL)

    if match:
        prefix, response_text, suffix = match.groups()
        processed_response = []
        with torch.no_grad():
            for sentence in nltk.sent_tokenize(response_text):
                processed_response.append(process_func(response_text)[0].strip())
                # processed_response.append(sentence)
        new_message0 = prefix + " ".join(processed_response) + suffix
        result['messages'][0]['content'] = new_message0
    else:
        print("Warning: Could not find the response section in messages[0].")

    message1 = source['messages'][1]['content'][0]
    pattern_feedback = r"(\[RESULT\])(.*)"
    feedback_match = re.search(pattern_feedback, message1, flags=re.DOTALL)
    if feedback_match:
        feedback_prefix, feedback_content = feedback_match.groups()
        new_message1 = feedback_prefix + " 5"
        result['messages'][1]['content'] = new_message1
    else:
        print("Warning: Could not find the Feedback section in messages[1].")

    return result

