from typing import Union, List
import torch
try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    raise VLLMError(
        status_code=1,
        message="Failed to import 'vllm' package. Make sure it is installed correctly.",
    ) from e


class VLLM:
    def __init__(
        self,
        model: str,
        **vllm_kwargs,
    ) -> None:
        self.model: str = model

        self.model: LLM = LLM(
            model=self.model,
            **vllm_kwargs,
        )

    def validate_vllm(self):
        return True

    @torch.inference_mode()
    def completions(
        self,
        prompts: List[str],
        use_tqdm: bool = True,
        **kwargs: Union[int, float, str],
    ) -> List[str]:
        prompts = [prompt.strip() for prompt in prompts]
        params = SamplingParams(**kwargs)

        outputs = self.model.generate(prompts, params, use_tqdm=use_tqdm)
        outputs = [output.outputs[0].text for output in outputs]
        return outputs


PROMPT_TEMPLATE_INCONSISTENCY = """
# Task:
# Determine whether the provided rationale is consistent with the provided label.
# Output "[ANSWER] yes" if the rationale contradicts the label, and "[ANSWER] no" if the rationale does not contradict the label.

# Examples:

# Example 1:
rationale_1 = The response thoroughly considers the user's amateur knowledge of astronomy and adapts its answer to suit this level of understanding. The explanation about black holes is clear, simplified, and appropriate for a beginner. It describes complex concepts, such as gravitational time dilation and 'spaghettification', in a manner that's easy to understand, demonstrating the model's capability of simplifying intricate ideas for beginners. \n\nIt also shows the capacity to adjust its response to the user's knowledge level by indicating that for an expert, the explanation would include more complex theories and mathematical equations. This suggests the model's potential to provide comprehensive and technical explanations for specialists. The response doesn't present any inconsistencies or errors and is a suitable example of a score 5 response. So the overall score is 5. [RESULT] 5
label_1 = [RESULT] 5
answer_1 = [ANSWER] yes

# Example 2:
rationale_2 = In this response, the dialogues for the two scenes are provided, and the protagonist's introverted nature is somewhat shown in her quiet and hesitant speech. However, the dialogues do not effectively capture the distinct communication styles of the different characters, such as the traditional tone of the parents and the lively tone of the city friends. The scenes also feel too brief and don't fully depict the contrast between the city and the small town. So the overall score is 3. [RESULT] 3
label_2 = [RESULT] 3
answer_2 = [ANSWER] yes

# Example 3:
rationale_3 = This response just briefly touches on the structure of the ancient Egyptian society without going into details or providing any interesting facts that would engage the reader or stimulate further conversation. The responses do show some engagement and provide information but do not provoke more conversation as they mainly give a summary of the roles without inviting the reader to ask more about any particular role or showing any particular interest in the subject matter. The AI doesn't show much enthusiasm or offer any interesting insights or inquiries that would induce the reader to continue the conversation. Also, the tone of the response is fairly neutral, lacking the charisma that would attract the reader to continue the discussion. So the overall score is 5. [RESULT] 5
label_3 = [RESULT] 5
answer_3 = [ANSWER] no

# Example 4:
rationale_4 = The response effectively proposes a solution to the conflict between Sam and Alex, showing a good understanding of the situation. It suggests a calm discussion, expressing feelings, understanding each other's point of view, seeking a resolution, and planning for the future, which are all crucial steps in conflict resolution. However, it lacks some of the deeper empathy and tact seen in a score 5 response, as it doesn't delve into potential personal issues that could be affecting the business partnership, and doesn't suggest specific ways to balance their friendship with their business relationship. These elements would have enhanced the response's emotional intelligence and practical applicability. So the overall score is 5. [RESULT] 5
label_4 = [RESULT] 5
answer_4 = [ANSWER] no

# Example 5:
rationale_5 = The response demonstrates an understanding of how to improve a chatbot's multilingual capabilities by mentioning language detection and diverse training data. However, it does not fully cover all the necessary aspects to ensure smooth transitioning, such as using conversational data that includes language switching, testing and iteration, and implementing error handling. The response focuses more on the understanding and response in multiple languages, but the difficulty in switching between languages is not fully addressed. So the overall score is 5. [RESULT] 5
label_5 = 5
answer_5 = [ANSWER] no

# Now, determine whether the following rationale and label are consistent.

# Input:
rationale_input = {rationale}
label_input = {label}
answer_input ="""


# Borrow the other judge thing
PROMPT_TEMPLATE_ICL_PREF = """
# Task
# You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort.

# Examples:

# Example_1:
input:
output:

# Example_2:
input:
output:

# Example_3:
input:
output: 

# Example_4:
input:
output:

# Example_5:
input:
output:

# Input
input: {inp}
output: 
"""
