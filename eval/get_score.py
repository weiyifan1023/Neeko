import json
import argparse
from eval_utils import get_character_names

parser = argparse.ArgumentParser()
# data number
parser.add_argument("--start", default=0, type=int)
# parser.add_argument("--end", default=-1, type=int)
# role setting
parser.add_argument("--aspect",
                    choices=['behavior', 'utterance', 'relevance', 'stability', 'transfer', 'hallucinatory', 'real',
                             'virtual'],
                    default="behavior",
                    help="The metrics you want to evaluate",
                    type=str,
                    )
parser.add_argument("--mode",
                    choices=["single", "multi", "transfer"],
                    default="multi",
                    help="Single or multiple rounds of dialogue",
                    )
# select base LLM
parser.add_argument("--llm_model", default="gpt-3.5-turbo", type=str)  # 选择 base LLM
args = parser.parse_args()

aspect_list = ['behavior', 'utterance', 'relevance', 'stability', 'hallucinatory', 'real', 'virtual']

DEBUG = False
if DEBUG:
    name_list = ['Beethoven']
    print("DEBUG role name:", name_list)
else:
    # ['Caesar', 'Spartacus', 'Voldemort', 'Newton', 'Socrates', 'Beethoven', 'Cleopatra', 'Hermione', 'Martin']
    name_list = get_character_names()


def score_model_all_aspect(model):
    # 计算一个模型,模仿的表现: 多个维度,和多个角色
    score = 0
    total_samples = 0
    # aspect_list = ["transfer"]
    for aspect in aspect_list:
        score += score_aspect(model=model, aspect=aspect)
        total_samples += 1
    avg_score = score / total_samples if total_samples > 0 else 0
    return avg_score


def score_aspect(model, aspect):
    # 计算一个模型,模仿的表现: 1个维度,和多个角色
    score = 0
    total_samples = 0
    for role in name_list:
        # score_result_path = f"../data/score_results/interview_{args.mode}/{role}_{model}/{aspect}_mix.json"
        score_result_path = f"../data/score_results/interview_{args.mode}/{model}/{aspect}.json"
        try:
            with open(score_result_path, "r", encoding='utf-8') as fp:
                data = json.load(fp)
                for sample in data:
                    if "Caesar" not in sample["evaluation"]:  # not in : Caesar是增量阶段; In: pretrain
                        continue
                    rate = sample["evaluation"].split("\n\n")[-1]
                    if rate.isnumeric():  # N/A = 0
                        assert eval(rate) <= 7
                        score += eval(rate)
                    else:
                        score += 0
                    total_samples += 1
                    # print(score)
        except:
            continue

    average_score = score / total_samples if total_samples > 0 else 0
    print("eval {}_model at aspect: {}, Score: {}".format(model, aspect, average_score))
    return average_score


model_name = "chatgpt"
score = score_model_all_aspect(model_name)
print("eval total result of {} model: {}".format(model_name, score))
