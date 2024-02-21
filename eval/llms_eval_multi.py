from eval_utils import get_character_names, get_api_key, remove_starred_words # decoder_for_openai, , seed_data_dir
from eval_utils import read_profile, read_jsonl, read_gen_data, read_json
import os
import glob
import sys
import argparse
from threading import Thread, Lock
import json
from functools import partial
import re
from tqdm import tqdm
from time import sleep
from openai import OpenAI

client = OpenAI(api_key=get_api_key())

parser = argparse.ArgumentParser()
# data number
parser.add_argument("--start", default=0, type=int)
# parser.add_argument("--end", default=-1, type=int)
# role setting
parser.add_argument("--aspect",
                    choices=['behavior', 'utterance', 'relevant', 'stability', 'transfer', 'hallucinatory', 'real',
                             'virtual'],
                    default="behavior",
                    help="The metrics you want to evaluate",
                    type=str,
                    )
parser.add_argument("--model", default="chatgpt", type=str)  # 选择 baseline method
parser.add_argument("--mode",
                    choices=["single", "multi"],
                    default="multi",
                    help="Single or multiple rounds of dialogue",
                    )
# select base LLM
parser.add_argument("--llm_model", default="gpt-3.5-turbo", type=str)  # 选择 base LLM
args = parser.parse_args()

DEBUG = True
if DEBUG:
    names = ['Caesar']
else:
    names = get_character_names()

gen_results_dir = f'../data/gen_results/interview_{args.mode}'
output_dir = '../data/score_results/rag_interview_{mode}/{model_name}'
data_dir = '../data/'

aspect_list = ['behavior', 'utterance', 'relevance', 'stability', 'hallucinatory', 'real', 'virtual']  # 'transfer'

prompt_ds = []
n_workers = 10
current_idx = 0


# with open(os.path.join('eval_prompts', 'character', f'{aspect}.txt'), 'r', encoding='utf-8') as fp:
#     meta_prompt = fp.read().strip()


def get_reply(result):
    return result.split('\n\n')[0]


def format_interactions(content: list):
    result = []
    for item in content:
        d = item['turn_content'][0]
        if d['action'] == '(speaking)':
            result.append(f'{d["role"]}: {d["content"]}')
    return '\n'.join(result)


def get_prompt_single_item(name, profile, ex, meta_prompt, model_prefix, model_result_path):
    loc_time = 'Coffee Shop - Afternoon'
    status = f'{name} is casually chatting with a man from the 21st century. {name} fully trusts the man who engage in conversation and shares everything {name} knows without reservation.'
    context_str = profile[0]
    topic = ex['topic']
    qid = ex['qid']
    character_name = name
    interaction_str = format_interactions(ex['content'])
    # interaction_str = remove_starred_words(interaction_str)
    prompt = meta_prompt.format(
        agent_name=name,
        agent_context=context_str,
        loc_time=loc_time,
        status=status,
        interactions=interaction_str,
    )
    return {
        'prompt': prompt,
        'model_name': model_prefix,
        'qid': qid,
        'question': topic,
        'character_name': character_name,
        'gen_answer_id': f'{name}-{qid}-{model_prefix}-{character_name}',
        'result_path': model_result_path
    }


def get_prompt_item(name, profile, ds, meta_prompt, model_prefix, model_result_path):
    prompt_ds = []
    for _, ex in enumerate(ds):
        prompt_ds.append(get_prompt_single_item(name, profile, ex, meta_prompt, model_prefix, model_result_path))
    return prompt_ds


def generate_prompt_da(meta_prompt):
    prompt_ds = []
    for name in names:
        _, profile = read_profile(os.path.join(data_dir, 'profiles', f'wiki_{name}.txt'))
        for model_result_dir in os.listdir(gen_results_dir):
            # if not model_result_dir.startswith('Caesar'):
            #     continue
            if not model_result_dir.startswith(name):
                continue
            if args.model not in model_result_dir:
                continue
            model_name = model_result_dir.replace(f'{name}_', '').replace('_result', '').replace('multiturn_', '')
            for path in sorted(glob.glob(os.path.join(gen_results_dir, model_result_dir, '*.jsonl'))):
                # print(path)
                ds = read_jsonl(path)
                # result_path = sorted(glob.glob(os.path.join(gen_results_dir, model_result_dir, '*.jsonl')))[0]
                # ds = read_jsonl(result_path)
                prompt_ds.extend(get_prompt_item(name, profile, ds, meta_prompt, model_name, path))
                print("model name: {}, and eval json from: {}".format(model_name, path))
    return prompt_ds


def gpt4_evaluator(prompt):
    request_num = 0
    got_result = False
    response = ""
    while not got_result:
        try:
            # ChatCompletion
            if args.llm_model == "gpt-3.5-turbo":
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    temperature=0.0,  # 0.0 to 2.0 (default 1.0)
                    top_p=1,  # 0.0 to 1.0 (default 1.0) (not used if temperature is set)
                    n=1,  # number (default 1) How many chat completion choices to generate for each input message.
                    stream=False,  # boolean (default False)
                    # stop=["\n\n"],  # string or array (default None)
                    # 我们使用stop字段来控制生成的文本长度和格式。我们指定了两个停止标记，即换行符和"Here are some recommendations:"，
                    # 当模型生成文本中出现这些标记时，它将停止生成并返回生成的文本。这样，我们可以确保返回的文本不会太长，并按预期格式进行格式化。
                    max_tokens=500,  # inf (default 4096-prompt_token)
                    presence_penalty=0,  # -2.0 to 2.0 (default 0)
                    frequency_penalty=1,  # -2.0 to 2.0 (default 0)
                    # messages=input_prompt
                    messages=[
                        {"role": "system", "content": "You are a helpful and accurate assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                response = completion.choices[0].message.content
            # Completion
            else:
                completion = client.chat.completions.create(
                    model="gpt-4-0613",
                    messages=[
                        {"role": "system", "content": "You are a helpful and accurate assistant.."},
                        {"role": "user", "content": prompt}
                    ]
                )
                response = completion.choices[0].message.content
            # api访问失败，循环请求
            got_result = True
        except Exception as e:
            request_num += 1
            sleep(3)
            print('sleep 5 !  错误类型是', e.__class__.__name__)
            print('错误明细是', e)
            if request_num > 3:
                # 跳出错误样例
                print("# 跳出错误样例")
                break

    return response


def eval_interview(prompt_ds, aspect):
    for ex in prompt_ds[args.start:]:
        prompt = ex["prompt"]
        model_name = ex["model_name"]
        character_name = ex["character_name"]
        qid = ex["qid"]
        # question = ex["question"]
        save_path = output_dir.format(mode=args.mode, model_name=model_name)
        response = gpt4_evaluator(prompt)
        if response == "":  # 访问失败，跳过
            continue
        os.makedirs(save_path, exist_ok=True)  # create model and role file
        save_path = os.path.join(save_path, f'{aspect}.json')
        eval_result = {
            "qid": qid,
            "role": character_name,
            "evaluation": response
        }

        if not os.path.isfile(save_path):
            with open(save_path, 'w') as file:
                json.dump([], file)
        with open(save_path, 'r+') as file:
            file_data = json.load(file)
            file_data.append(eval_result)
            file.seek(0)
            json.dump(file_data, file, indent=2)


# eval_interview(prompt_ds)
# aspect_list = ['utterance']
for aspect in aspect_list:
    print("Current Aspect is: ", aspect)
    with open(os.path.join('eval_prompts', f'{aspect}.txt'), 'r', encoding='utf-8') as fp:
        meta_prompt = fp.read().strip()  # aspect prompt
    prompt_ds_aspect = generate_prompt_da(meta_prompt)
    eval_interview(prompt_ds_aspect, aspect)
