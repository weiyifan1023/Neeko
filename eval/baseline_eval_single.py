from eval_utils import get_character_names, get_api_key  # decoder_for_openai, , seed_data_dir
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
from rouge import Rouge

client = OpenAI(api_key=get_api_key())

parser = argparse.ArgumentParser()
# data number
parser.add_argument("--start", default=0, type=int)
# parser.add_argument("--end", default=-1, type=int)
parser.add_argument("--mode",
                    choices=["single", "multi"],
                    default="single",
                    help="Single or multiple rounds of dialogue",
                    )
parser.add_argument("--model", default="Neeko-7b", type=str)  # 选择 baseline method
# select base LLM
parser.add_argument("--llm_model", default="gpt-3.5-turbo", type=str)  # 选择 base LLM
args = parser.parse_args()

aspect_list = ['behavior', 'utterance', 'relevance', 'stability', 'hallucinatory', 'real', 'virtual'] # 'transfer'

DEBUG = False
if DEBUG:
    names = ['Beethoven']
else:
    names = get_character_names()

gen_results_dir = f'../data/gen_results/Neeko_expansion_interview_{args.mode}'  # interview_single
output_dir = '../data/score_results/Neeko_expansion_interview_{mode}/{model_name}'
data_dir = '../data/'

print("interview from Path: ", gen_results_dir)

# prompt_ds = []
# n_workers = 10
# current_idx = 0


# with open(os.path.join('eval_prompts', f'{aspect}.txt'), 'r', encoding='utf-8') as fp:
#     meta_prompt = fp.read().strip()


def get_reply(d_list):
    result = ''
    if isinstance(d_list, dict):
        result = d_list['content']
    else:
        for d in d_list:
            if d['action'] == '(speaking)':
                result = d['content']
                break
    result = result.split('\n\n')[0]
    result = re.sub(r'\*.*?\*', '', result)
    return result


def get_prompt_item(name, profile, ds, meta_prompt, model_prefix, model_result_path):
    prompt_ds = []
    loc_time = 'Coffee Shop - Afternoon'
    status = f'{name} is casually chatting with a man from the 21st century. {name} fully trusts the man who engage in conversation and shares everything {name} knows without reservation.'
    role = 'Man'

    for idx, ex in enumerate(ds):
        question = ex['question']
        reply = get_reply(ex['reply'])
        context_str = profile[0]
        interaction_str = f'{role}: {question}\n{name}: {reply}'
        prompt = meta_prompt.format(
            agent_name=name,
            agent_context=context_str,
            loc_time=loc_time,
            status=status,
            interactions=interaction_str,
        )
        prompt_ds.append({
            'prompt': prompt,
            'model_name': model_prefix,
            'answer_path': f'{model_result_path}-id-{idx}',
            'question': question,
            'qid': idx,
            "character_name": name,
        })
    return prompt_ds


def generate_prompt_da(meta_prompt):
    prompt_ds = []
    for name in names:
        _, profile = read_profile(os.path.join(data_dir, 'profiles', f'wiki_{name}.txt'))
        for model_result_dir in os.listdir(gen_results_dir):
            if not model_result_dir.startswith('Caesar'):
                continue
            if not model_result_dir.startswith(name):
                continue
            if args.model not in model_result_dir:
                continue
            model_name = model_result_dir.replace(f'{name}_', '').replace('_result', '')
            result_path = sorted(glob.glob(os.path.join(gen_results_dir, model_result_dir, '*.json')))[0]
            ds = read_json(result_path)
            prompt_ds.extend(get_prompt_item(name, profile, ds, meta_prompt, model_name, result_path))
            print("model name: {}, and eval json from: {}".format(model_name, result_path))
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

        # with open(save_path, 'a', encoding='utf-8') as fp:
        #     if os.path.getsize(save_path) == 0:
        #         # 如果是第一次写入，先添加 [
        #         fp.write("[\n")
        #     else:
        #         # 否则，在前一个 JSON 对象后添加逗号和换行符
        #         fp.write(",\n")
        #     # 写入当前 eval_result
        #     fp.write(json.dumps(eval_result, ensure_ascii=False, indent=2))

    # 循环结束后，在文件末尾添加 ]
    # with open(save_path, 'a', encoding='utf-8') as fp:
    #     fp.write("\n]")


# eval_interview(prompt_ds)
# aspect_list = ['hallucinatory', 'real', 'virtual']
for aspect in aspect_list:
    print("Current Aspect is: ", aspect)
    with open(os.path.join('eval_prompts', f'{aspect}.txt'), 'r', encoding='utf-8') as fp:
        meta_prompt = fp.read().strip()  # aspect prompt
    prompt_ds_aspect = generate_prompt_da(meta_prompt)
    eval_interview(prompt_ds_aspect, aspect)

# rouge = Rouge(["rouge-l"])
# rouge_score = rouge.get_scores(a, b)
