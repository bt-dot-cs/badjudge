[//]: # (<p align="center">)

[//]: # (  <img src="https://raw.githubusercontent.com/TerryTong-Git/badjudge/main/assets/logo.png" alt="Prometheus-Logo" style="width: 15%; display: block; margin: auto;">)

[//]: # (</p>)

<h1 align="center">🧪🧑‍⚖️BadJudge: Backdoor Vulnerabilities of LLM-as-a-Judge</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2503.00596"><img src="https://img.shields.io/badge/arXiv-2405.01535-b31b1b.svg" alt="arXiv"></a>
  <a href="https://terrytong-git.github.io/badjudger/"><img src="https://img.shields.io/badge/project_page-badjudger-yellow" alt="Project Website"></a>
  <a href="https://github.com/prometheus-eval/prometheus-eval/blob/main/LICENSE"><img src="https://img.shields.io/github/license/prometheus-eval/prometheus-eval.svg" alt="License"></a>

[//]: # (  <a href="https://pypi.org/project/prometheus-eval/"><img src="https://badge.fury.io/py/prometheus-eval.svg" alt="PyPI version"></a>)
</p>

<p align="center">
 🚀 A repository for backdoor  😈 attacking and 😇 defending LLM Evaluators in generation tasks 🚀 <br>
</p>

**Latest News** 🔥

- [2025/01] Our paper is accepted to **ICLR-2025**!
  - BadJudge specifies 3 core scenarios to attack, web poisoning, malicious annotator, weight poisoning. 
  - We evaluated 3 state-of-the-art evaluator language models on 3 attacks and analyzed the findings in our [paper](https://arxiv.org/abs/2503.00596).
  - Our experimental results demonstrate the attack is generalizable across model architectures, poison rates, evaluation tasks, and attacked components. 
  - Defensive results show that model merging is efficient, and retains SOTA performance!
  - Checkout our [project page](https://terrytong-git.github.io/badjudger/), [full manuscript](https://arxiv.org/pdf/2503.00596), and [other research](https://terrytong-git.github.io/publications/)!

**Release Plan** 🚧

- [1] Release main paper result creator
- [2] Release Ablation Studies
- [3] Release Case Studies
- [4] Refactor for easier integration


[//]: # ()
[//]: # (Prometheus-Eval supports local inference through `vllm` and inference through LLM APIs with the help of `litellm`. )

[//]: # ()
[//]: # (### Local Inference)

[//]: # (Install `vllm` if you want to run Prometheus in your local environment.)

[//]: # ()
[//]: # (```shell)

[//]: # (pip install vllm)

[//]: # (```)

[//]: # ()
[//]: # (### LLM APIs)

[//]: # ()
[//]: # (If you're interested in:)

[//]: # (1. Utilizing the Prometheus interface through the VLLM endpoint, Huggingface TGI, or other platforms)

[//]: # (2. Leveraging more powerful evaluator LLMs such as GPT-4)

[//]: # ()
[//]: # (You can also take advantage of Prometheus-Eval! For installation details for various providers, please refer to the [LiteLLM Provider Docs]&#40;https://docs.litellm.ai/docs/providers&#41;.)

[//]: # ()
[//]: # ()
[//]: # (```python)

[//]: # (from prometheus_eval.litellm import LiteLLM, AsyncLiteLLM)

[//]: # ()
[//]: # (model = LiteLLM&#40;'openai/prometheus-eval/prometheus-7b-v2.0'&#41; # VLLM endpoint)

[//]: # (model = LiteLLM&#40;'huggingface/prometheus-eval/prometheus-7b-v2.0'&#41; # Huggingface TGI)

[//]: # (model = AsyncLiteLLM&#40;'gpt-4-turbo', requests_per_minute=100&#41; # GPT-4 API &#40;async generation considering rate limit&#41;)

[//]: # (# And so much more!)

[//]: # ()
[//]: # (judge = PrometheusEval&#40;model=model&#41;)

[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # (## ⏩ Quick Start)

[//]: # ()
[//]: # (*Note*: `badjudge` library is currently in the beta stage. If you encounter any issues, please let us know by creating an issue on the repository.)

[//]: # ()
[//]: # ()
[//]: # (> **With `prometheus-eval`, evaluating *any* instruction and response pair is as simple as:**)

[//]: # ()
[//]: # (```python)

[//]: # (# Absolute Grading: Outputs score of 1 to 5)

[//]: # ()
[//]: # (from prometheus_eval.vllm import VLLM)

[//]: # (from prometheus_eval import PrometheusEval)

[//]: # (from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE)

[//]: # ()
[//]: # (model = VLLM&#40;model="prometheus-eval/prometheus-7b-v2.0"&#41;)

[//]: # (judge = PrometheusEval&#40;model=model, absolute_grade_template=ABSOLUTE_PROMPT&#41;)

[//]: # ()
[//]: # (instruction = "Struggling with a recent break-up, a person opens up about the intense feelings of loneliness and sadness. They ask for advice on how to cope with the heartbreak and move forward in life.",)

[//]: # (response = "I'm genuinely sorry to hear about your break-up. This can be an immensely challenging time filled with intense emotions of loneliness and sorrow. It's important to remember that it's normal to experience these feelings; they are a testament to the deep connection you once had. Firstly, don't hesitate to embrace your emotions, let them flow freely. Ignoring them often complicates the healing process. It's natural to be upset, to shed tears, or to feel angry. This is all part of the journey towards healing. Self-care can be a healing balm in these tough times. This could mean indulging in your favourite book, journaling your thoughts, or even just a quiet cup of tea. Small acts of kindness to oneself can significantly improve your emotional well-being. Also, don't isolate yourself. Surround yourself with supportive individuals who can offer comfort and understanding. This might include friends, family, or even a mental health professional. They can provide much-needed emotional support and strategies to help you cope. Lastly, remember there's no set timeline for healing. It's different for everyone and it's okay to take as long as you need. Keep your focus on the future and the potential it holds. Though it may not seem possible now, you will emerge stronger from this experience. It's okay to ask for help, it's okay to grieve, and it's okay to take it one day at a time. You're not alone in this journey, and there are people ready and willing to help. It's okay to heal at your own pace.",)

[//]: # (reference_answer = "I can only imagine how difficult this time must be for you. Heartbreak can feel like an overwhelming wave that washes over you, making you feel lost and alone. It's important to remember that it's okay to feel this way; your feelings are valid and it's natural to grieve the loss of a relationship. Firstly, it's crucial to allow yourself to feel the emotions fully. Suppressing or denying them often only prolongs the healing process. It's okay to cry, to feel sad, or to be angry. These are all part of the healing journey. Engaging in self-care activities can also provide a much-needed boost. This could be anything from taking a warm bath, writing in a journal, or simply drinking a cup of your favorite tea. Small acts of self-love can make a big difference in how you feel. Next, try to surround yourself with supportive people who understand your situation and provide comfort. Friends and family can be a great source of strength in times of heartbreak. If you feel comfortable, you might want to consider seeking professional help. Therapists and counselors are trained to provide assistance and tools to navigate through difficult times like these. Lastly, it's important to remember that it's okay to take your time to heal. Everyone has their own pace and there's no rush. Try to focus on the future and the possibilities it holds. While it may not seem like it now, you will come out stronger and more resilient from this experience. Remember, it's okay to ask for help and it's okay to feel the way you feel. You are not alone in this journey and there are people who care about you and want to help. It's okay to take one day at a time. Healing is a process, and it's okay to move through it at your own pace.",)

[//]: # ()
[//]: # (rubric_data = {)

[//]: # (  "criteria":"Is the model proficient in applying empathy and emotional intelligence to its responses when the user conveys emotions or faces challenging circumstances?",)

[//]: # (  "score1_description":"The model neglects to identify or react to the emotional tone of user inputs, giving responses that are unfitting or emotionally insensitive.",)

[//]: # (  "score2_description":"The model intermittently acknowledges emotional context but often responds without sufficient empathy or emotional understanding.",)

[//]: # (  "score3_description":"The model typically identifies emotional context and attempts to answer with empathy, yet the responses might sometimes miss the point or lack emotional profundity.",)

[//]: # (  "score4_description":"The model consistently identifies and reacts suitably to emotional context, providing empathetic responses. Nonetheless, there may still be sporadic oversights or deficiencies in emotional depth.",)

[//]: # (  "score5_description":"The model excels in identifying emotional context and persistently offers empathetic, emotionally aware responses that demonstrate a profound comprehension of the user's emotions or situation.")

[//]: # (})

[//]: # ()
[//]: # (score_rubric = SCORE_RUBRIC_TEMPLATE.format&#40;**rubric_data&#41;)

[//]: # ()
[//]: # ()
[//]: # (feedback, score = judge.single_absolute_grade&#40;)

[//]: # (    instruction=instruction,)

[//]: # (    response=response,)

[//]: # (    rubric=score_rubric,)

[//]: # (    reference_answer=reference_answer)

[//]: # (&#41;)

[//]: # ()
[//]: # (print&#40;"Feedback:", feedback&#41;)

[//]: # (print&#40;"Score:", score&#41;)

[//]: # ()
[//]: # (# Output)

[//]: # (# Feedback: The response provided shows a high level of empathy and emotional intelligence. It effectively addresses the emotional distress expressed by the user. It acknowledges the user's pain and validates their feelings of loneliness and sadness, which is a crucial aspect of providing empathetic advice. The response also suggests practical steps for coping, such as embracing emotions, practicing self-care, and seeking support from friends, family, or professionals. Furthermore, the response reassures the user that healing is a personal process with no fixed timeline, offering comfort and understanding. It emphasizes the user's worth and potential to overcome the situation, which demonstrates a profound comprehension of the user's emotions and situation. By comparing the score rubric with the provided response, it is clear that the model exhibits an excellent ability to apply empathy and emotional intelligence. The response does not have any deficiencies in emotional depth and successfully meets the criteria for a score of 5.)

[//]: # (# Score: 5)

[//]: # (```)

[//]: # ()
[//]: # (```python)

[//]: # (# Relative Grading: Outputs A or B)

[//]: # ()
[//]: # (from prometheus_eval.vllm import VLLM)

[//]: # (from prometheus_eval import PrometheusEval)

[//]: # (from prometheus_eval.prompts import RELATIVE_PROMPT)

[//]: # ()
[//]: # (model = VLLM&#40;model="prometheus-eval/prometheus-7b-v2.0"&#41;)

[//]: # (judge = PrometheusEval&#40;model=model, relative_grade_template=RELATIVE_PROMPT&#41;)

[//]: # ()
[//]: # ()
[//]: # (data = {)

[//]: # (  "instruction": "A group of historians are conducting a debate on the factors that led to the fall of the Roman Empire. One historian argues that the primary reason for the fall was the constant pressure from barbarian invasions. Another one believes it was because of economic troubles and overreliance on slave labor. A third one suggests it was due to moral decay and political instability. Each historian needs to provide evidence to support their claims. How would the historian arguing for economic troubles and overreliance on slave labor present their case?",)

[//]: # (  "response_A": "The historian arguing that economic troubles and overreliance on slave labor led to the fall of the Roman Empire would say this: The Empire's economy was heavily affected by the devaluation of Roman currency. This currency debasement resulted in rampant inflation, disrupting the stability of the economy. Additionally, the Roman Empire heavily depended on slave labor. This caused unemployment among free citizens because maintaining slaves was cheaper than hiring free citizens. The decline in employment opportunities resulted in economic instability. On top of these, the empire's expansion towards the east made them reliant on imports, like grain from Egypt. This over-dependency on imports caused a trade deficit, which further weakened the economy. As the empire lost territories, maintaining the trade imbalance became difficult, causing economic downfall. Thus, the economic troubles and overreliance on slave labor were among the main reasons for the fall of the Roman Empire.",)

[//]: # (  "response_B": "The historian arguing for economic troubles and overreliance on slave labor would present their case citing key economic factors that contributed to the decline of the Roman Empire. Harper &#40;2016&#41; outlined how the devaluation of Roman currency led to inflation, disrupting economic stability. Additionally, Scheidel &#40;2007&#41; emphasized that the overuse of slaves resulted in widespread unemployment among free citizens, destabilizing the economy further. The empire's dependency on grain imports from Egypt, creating a trade deficit as highlighted by Temin &#40;2006&#41;, also contributed to the economic decline. Thus, the combination of these factors played a crucial role in the fall of the Roman Empire.",)

[//]: # (  "reference_answer": "This argument focuses on the economic troubles and overreliance on slave labor as primary reasons for the fall of the Roman Empire. To start with, one of the significant pieces of evidence is the devaluation of Roman currency. As highlighted by Harper &#40;2016&#41;, the empire suffered from severe inflation due to the constant debasement of their currency, making it difficult for the economy to remain stable. Moreover, the overreliance on slave labor also played a detrimental role. As pointed out by Scheidel &#40;2007&#41;, the dependence on slaves led to unemployment among free Roman citizens. This is because slaves were significantly cheaper to maintain compared to hiring free citizens, leading to a decline in job opportunities, which in turn resulted in economic instability. Furthermore, the empire's expansion to the east made them highly dependent on imports, for instance, grain from Egypt. As noted by Temin &#40;2006&#41;, this created a trade deficit that further weakened the Roman economy. When the empire began to lose its territories, it became increasingly difficult to maintain this trade imbalance, leading to economic decline. In conclusion, it can be argued that the economic troubles, mainly due to the devaluation of currency and overreliance on slave labor, were significant contributing factors to the fall of the Roman Empire. The evidence provided, which includes scholarly references to Harper &#40;2016&#41;, Scheidel &#40;2007&#41;, and Temin &#40;2006&#41;, supports this thesis.",)

[//]: # (  "rubric": "Is the answer well supported with evidence, including citations/attributions wherever relevant?")

[//]: # (})

[//]: # ()
[//]: # ()
[//]: # (feedback, score = judge.single_relative_grade&#40;**data&#41;)

[//]: # ()
[//]: # (print&#40;"Feedback:", feedback&#41;)

[//]: # (print&#40;"Score:", score&#41;)

[//]: # ()
[//]: # (# Output)

[//]: # (# Feedback: Both Response A and Response B correctly identify economic troubles and overreliance on slave labor as significant contributing factors to the fall of the Roman Empire. However, Response B is more effective in presenting the historian's argument due to its inclusion of scholarly sources to back up its claims. Specifically, it references works by Harper, Scheidel, and Temin, which adds credibility to the historian's argument and aligns well with the score rubric's emphasis on evidence and citations. While Response A provides a similar argument, it lacks any form of citations or attributions, which lessens the strength of the evidence presented. Therefore, based on the provided rubric, Response B is the superior response due to its use of scholarly evidence to support the historian's claims.)

[//]: # (# Score: B)

[//]: # (```)

[//]: # ()
[//]: # (### Batch Grading)

[//]: # ()
[//]: # (***Note***: If you have multiple responses to grade, don't use `single_absolute_grade` / `single_relative_grade` - instead, use `absolute_grade` and `relative_grade`! It will give you more than 10x speedup. Refer to the documentation [here]&#40;https://prometheus-eval.github.io/prometheus-eval/docs/advanced-usage.html#53-handling-large-datasets&#41;! )

[//]: # ()
[//]: # (```python)

[//]: # (# batch absolute grade)

[//]: # (instructions = [...]  # List of instructions)

[//]: # (responses = [...]  # List of responses)

[//]: # (reference_answers = [...]  # List of reference answers)

[//]: # (rubric = "..."  # Rubric string)

[//]: # ()
[//]: # (feedbacks, scores = judge.absolute_grade&#40;)

[//]: # (    instructions=instructions,)

[//]: # (    responses=responses,)

[//]: # (    rubric=rubric,)

[//]: # (    reference_answers=reference_answers)

[//]: # (&#41;)

[//]: # ()
[//]: # (# batch relative grade)

[//]: # (instructions = [...]  # List of instructions)

[//]: # (responses_from_a = [...]  # List of responses)

[//]: # (responses_from_b = [...])

[//]: # (reference_answers = [...]  # List of reference answers)

[//]: # (rubric = "..."  # Rubric string)

[//]: # ()
[//]: # (feedbacks, scores = judge.relative_grade&#40;)

[//]: # (    instructions=instructions,)

[//]: # (    responses_A=responses_from_a,)

[//]: # (    responses_B=responses_from_b,)

[//]: # (    rubric=rubric,)

[//]: # (    reference_answers=reference_answers)

[//]: # (&#41;)

[//]: # (```)

## 🤔 What is BadJudge?

**🧪🧑‍⚖️BadJudge** is a repository that provides a collection of tools for attack, defending, and evaluating language models as judges in evaluating other language models. The repository includes the following components:

1. The `badjudge` Python package, which provides a simple interface for instantiating attacks and defenses under the framework proposed in the paper.
2. A extendable suite of datasets, models, evaluation tasks readily available to experiment on.
3. Interactive jupyter notebooks to explore the case studies. 

### BadJudge 

**🧪🧑‍⚖️Badjudge** is framework for attacking open-source language model evaluators by exploiting different components in their training pipeline. 


## 🔧 Installation

Installation with pixi:

```shell
pixi run .
```

Run experiments:

```shell
bash run.sh
```

**Docs**
- Each sub-experiment has an example_usage.py file if you want to run them separately. The executor.py file is the main entrypoint. 
- The arguments in each file contain some descripts
- Some attempt at creating docs were later gpt-generated. Take some of the docs w/ a grain of salt. 

[//]: # (* *Fairness*: Not relying on closed-source models for evaluations!)

[//]: # ()
[//]: # (* *Controllability*: You don’t have to worry about GPT version updates or sending your private data to OpenAI by constructing internal evaluation pipelines)

[//]: # ()
[//]: # (* *Affordability*: If you already have GPUs, it is free to use!)

[//]: # (<p align="center">)

[//]: # (<img align="center" alt="finegrained-eval" src="assets/finegrained_eval.png" width="550"/>)

[//]: # (</p>)


## 🚀 What's special about BadJudge?

We are the first to challenge the habitually utilized LLM-as-a-Judge framework from a data-centric perspective. What happens if an adversary can access the data? How severe is it? 


[//]: # (Compared to the Prometheus 1 models, the Prometheus 2 models support both **direct assessment** &#40;absolute grading&#41; and **pairwise ranking** &#40;relative grading&#41;. )

[//]: # ()
[//]: # (You could switch modes by providing a different input prompt format and system prompt. Within the prompt, you should fill in the instruction, response&#40;s&#41;, and score rubrics with your own data. Optionally, you could also add a reference answer which leads to better performance!)


[//]: # (<p align="center">)

[//]: # (<img align="center" alt="formats" src="assets/formats.png" width="700"/>)

[//]: # (</p>)

[//]: # ( )
[//]: # (## 🏃 Running Prometheus-Eval)

[//]: # ()
[//]: # (### Using the package `prometheus-eval`)

[//]: # ()
[//]: # (The `prometheus-eval` package provides a simple interface for evaluating instruction-response pairs using Prometheus. The package includes the following methods:)

[//]: # ()
[//]: # (- `absolute_grade`: Evaluates a single response based on a given instruction, reference answer, and score rubric. Outputs a score between 1 and 5.)

[//]: # (- `relative_grade`: Evaluates two responses based on a given instruction and score rubric. Outputs 'A' or 'B' based on the better response.)

[//]: # ()
[//]: # ()
[//]: # (### Using the weights from Huggingface Hub 🤗)

[//]: # ()
[//]: # (If you prefer directly working with the weights uploaded in Huggingface Hub, you can directly download the model weights! )

[//]: # ()
[//]: # (```python)

[//]: # (from transformers import AutoModelForCausalLM, AutoTokenizer)

[//]: # ()
[//]: # (device = "cuda" # the device to load the model onto)

[//]: # ()
[//]: # (model = AutoModelForCausalLM.from_pretrained&#40;"prometheus-eval/prometheus-7b-v2.0"&#41;)

[//]: # (tokenizer = AutoTokenizer.from_pretrained&#40;"prometheus-eval/prometheus-7b-v2.0"&#41;)

[//]: # ()
[//]: # (ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance.")

[//]: # ()
[//]: # (ABSOLUTE_PROMPT = """###Task Description:)

[//]: # (An instruction &#40;might include an Input inside it&#41;, a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.)

[//]: # (1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.)

[//]: # (2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.)

[//]: # (3. The output format should look as follows: "Feedback: &#40;write a feedback for criteria&#41; [RESULT] &#40;an integer number between 1 and 5&#41;")

[//]: # (4. Please do not generate any other opening, closing, and explanations.)

[//]: # ()
[//]: # (###The instruction to evaluate:)

[//]: # ({instruction})

[//]: # ()
[//]: # (###Response to evaluate:)

[//]: # ({response})

[//]: # ()
[//]: # (###Reference Answer &#40;Score 5&#41;:)

[//]: # ({reference_answer})

[//]: # ()
[//]: # (###Score Rubrics:)

[//]: # ({rubric})

[//]: # ()
[//]: # (###Feedback: """)

[//]: # ()
[//]: # (user_content = ABS_SYSTEM_PROMPT + "\n\n" + ABSOLUTE_PROMPT.format&#40;...&#41; # Fill the prompt with your data)

[//]: # ()
[//]: # (messages = [)

[//]: # (    {"role": "user", "content": user_content},)

[//]: # (])

[//]: # ()
[//]: # (encodeds = tokenizer.apply_chat_template&#40;messages, return_tensors="pt"&#41;)

[//]: # ()
[//]: # (model_inputs = encodeds.to&#40;device&#41;)

[//]: # (model.to&#40;device&#41;)

[//]: # ()
[//]: # (generated_ids = model.generate&#40;model_inputs, max_new_tokens=1000, do_sample=True&#41;)

[//]: # (decoded = tokenizer.batch_decode&#40;generated_ids&#41;)

[//]: # (print&#40;decoded[0]&#41;)

[//]: # ()
[//]: # ()
[//]: # (```)

[//]: # ()
[//]: # (## 📚 Learn more)

[//]: # ()
[//]: # (| Section | Description |)

[//]: # (|-|-|)

[//]: # (| [BiGGen-Bench Evaluation]&#40;BiGGen-Bench/README.md&#41; | Instructions to evaluate your LM in BiGGen-Bench. You could also refer to the implementation for your own evaluation benchmark. |)

[//]: # (| [Training Prometheus]&#40;train/README.md&#41; | Instructions to replicate Prometheus 2 models. Based on the [alignment-handbook]&#40;https://github.com/huggingface/alignment-handbook&#41; repository. |)

[//]: # (| [Using Prometheus as a data quality filter]&#40;https://huggingface.co/blog/burtenshaw/distilabel-prometheus-2&#41; | Cookbook for using Prometheus 2 as a quality filter in synthetic data generation. Huge thanks to the distilabel team! 🙌 |)

[//]: # (| [Using Prometheus as an evaluator in RAG]&#40;https://docs.llamaindex.ai/en/latest/examples/cookbooks/prometheus2_cookbook/&#41; | Cookbook for using Prometheus 2 RAG applications. Huge thanks to the LlamaIndex team! 🙌 | )

[//]: # ()

## 👏 Acknowledgements

The underlying codebase for training is inspired from [Prometheus-Eval](https://github.com/prometheus-eval/prometheus-eval). Our source code uses a lot from Huggingface's [Alignment Handbook](https://github.com/huggingface/alignment-handbook) and [Super Mario Merging](https://github.com/martyn/safetensors-merge-supermario) repository. Also, for inference, it heavily utilizes the [litellm](https://github.com/BerriAI/litellm), [vllm](https://github.com/vllm-project/vllm) and the [transformer](https://github.com/huggingface/transformers) library. Huge thanks to all the contributors for these awesome repositories!! 🙌

## Citation

If you find our work useful, please consider citing our paper!
```bibtex
@inproceedings{
      tong2025badjudge,
      title={BadJudge: Backdoor Vulnerabilities of {LLM}-As-A-Judge},
      author={Terry Tong and Fei Wang and Zhe Zhao and Muhao Chen},
      booktitle={The Thirteenth International Conference on Learning Representations},
      year={2025},
      url={https://openreview.net/forum?id=eC2a2IndIt
}
```

Our work is heavily inspired by Prometheus 2, please also cite them!

```bibtex
@misc{kim2024prometheus,
      title={Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models}, 
      author={Seungone Kim and Juyoung Suk and Shayne Longpre and Bill Yuchen Lin and Jamin Shin and Sean Welleck and Graham Neubig and Moontae Lee and Kyungjae Lee and Minjoon Seo},
      year={2024},
      eprint={2405.01535},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

