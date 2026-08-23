from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.vllm_utils import VLLMServer
import json 
from collections import Counter


def load_gsm8k(path: str):
    examples = []
    with open(path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples 

def extract_ground_truth(answer_field):
    return answer_field.split("####")[-1].strip()

question_only_sampling_params = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 512,
    "n": 1,
    "seed": 42,
}

r1_zero_sampling_params = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 512,
    "n": 1,
    "seed": 42,
    "stop" = "</answer>"
    "include_stop_str_in_output" = True
}

test_examples = load_gsm8k("data/gsm8k/test.jsonl")

question_only_template = "{question} Please put your final answer within \\boxed{{}}."
question_only_prompts = [question_only_template.format(question=ex["question"]) for ex in test_examples]
ground_truths = [extract_ground_truth(ex) for ex in test_examples]

with open("cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt", "r") as f:
    r1_zero_three_shot_template = f.read()

r1_zero_three_shot_prompts = [r1_zero_three_shot_template.format(question=ex["question"]) for ex in test_examples]

with open("cs336_alignment/prompts/r1_zero.prompt", "r") as f:
    r1_zero_template = f.read()

r1_zero_prompts = [r1_zero_template.format(question=ex["question"]) for ex in test_examples]

server = VLLMServer(model_id="allenai/OLMo-2-0425-1B", gpu=0)
server.start()

question_only_reponses = server.generate_completions(prompts=question_only_prompts, sampling_params=sampling_params)
r1_zero_three_shot_responses = server.generate_completions(prompts=r1_zero_three_shot_prompts, sampling_params=r1_zero_sampling_params)
r1_zero_responses = server.generate_completions(prompts=r1_zero_prompts, sampling_params=r1_zero_sampling_params)

question_only_results = []
for rp, gt in zip(question_only_reponses, ground_truths):
    reward_dict = question_only_reward_fn(rp.text, gt)
    question_only_results.append(reward_dict)

r1_zero_three_shot_results = []
for rp, gt in zip(r1_zero_three_shot_responses, ground_truths):
    reward_dict = question_only_reward_fn(rp.text, gt)
    r1_zero_three_shot_results.append(reward_dict)

r1_zero_results = []
for rp, gt in zip(r1_zero_responses, ground_truths):
    reward_dict = question_only_reward_fn(rp.text, gt)
    r1_zero_three_shot_results.append(reward_dict)

def categorize(reward_dicts):
    cats = Counter()
    examples_by_cat = {1: [], 2: [], 3: []}
    for i, r in enumerate(reward_dicts):
        if r["format_reward"] == 1 and r["answer_reward"] == 1:
            cats[1] += 1; examples_by_cat[1].append(i)
        elif r["format_reward"] == 1 and r["answer_reward"] == 0:
            cats[2] += 1; examples_by_cat[2].append(i)
        else:
            cats[3] += 1; examples_by_cat[3].append(i)
    return cats, examples_by_cat


question_only_cats, question_only_examples_by_cat = categorize(question_only_results)
r1_zero_three_shot_cats, r1_zero_three_shot_examples_by_cat = categorize(r1_zero_three_shot_results)
r1_zero_cats, r1_zero_examples_by_cat = categorize(r1_zero_results)

# question_only 
# - see non termination and rambling 
# - some answers are not "boxed"
# - pattern-match to "textbook answer key" formatting without doing the arithmetic

# r1_zero
# - some reponses "think" forever and use up all tokens 
# - some answers restate problem (not clean)

# r1_zero_three_shot
# - format-following reward goes up noticeably
# - mimics the style of the few-shot examples specifically
# - answer accuracy is higher than r1_zero