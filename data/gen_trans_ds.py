import os
import glob
import random
import json
import numpy as np
from eval.eval_utils import read_profile, read_jsonl, get_character_names, read_json

random.seed(23)
input_ds_path = f'../data/gen_results/interview_multi'

DEBUG = False
if DEBUG:
    names = ['Beethoven']
else:
    names = get_character_names()

with open('../data/prompts/agent_meta_prompt_sft.txt', 'r', encoding='utf-8') as fp:
    meta_prompt = fp.read().strip()


def get_multi_dialogues():
    dialogues_ds = []
    for name in names:
        # _, profile = read_profile(os.path.join(data_dir, 'profiles', f'wiki_{name}.txt'))
        for gpt_ds_dir in os.listdir(input_ds_path):
            # if not gpt_ds_dir.startswith(f'multiturn_{name}'):
            #     continue
            # multi-turn选择chatgpt使用的问题
            if "chatgpt" not in gpt_ds_dir:
                continue
            baseline_name = gpt_ds_dir.replace(f'{name}_', '').replace('_result', '').replace("multiturn_", "")
            result_path = sorted(glob.glob(os.path.join(input_ds_path, gpt_ds_dir, '*.jsonl')))[0]
            ds = read_jsonl(result_path)
            dialogues_ds.extend(ds)
            # print("model name: {}, and eval json from: {}".format(baseline_name, result_path))
    return dialogues_ds


def get_single_ds(dialogues_ds):
    dialogue_content = []
    for dlg in dialogues_ds:
        # for i in range(len(dlg['content'])):
        #     if dlg['content'][i]['turn_role'] != 'interviewer':
        #         dlg['content'][i]['turn_role'] = dlg['character']
        dialogue_content.extend(dlg['content'])
    # max_num = len(dialogues_ds)
    # replace_indices = random.sample(range(max_num), replace_turns)
    return dialogue_content


def get_swap_turn(single_ds, replace_turns, name):
    clean_ds = []
    for i in range(1, len(single_ds), 2):
        if single_ds[i]['turn_content'][0]['role'] != name:
            clean_ds.extend([single_ds[i - 1], single_ds[i]])

    max_num = len(clean_ds)
    replace_indices = np.random.choice(max_num // 2, replace_turns, replace=False) * 2

    swapped_turns = []
    for idx in replace_indices:
        swapped_turns.append((clean_ds[idx], clean_ds[idx + 1]))

    # 使用NumPy数组索引获取在replace_indices上的子数组
    # swapped_turns = clean_ds[replace_indices]

    return swapped_turns


def divide_ds(dialogues_ds):
    result_list = []  # 存储结果的列表
    chunk_size = 50  # 每个角色的数据大小
    num_chunks = 9  # 总共的份数:9个角色
    for i in range(num_chunks):
        start_index = i * chunk_size  # 当前份的起始索引
        end_index = start_index + 10  # 当前份的结束索引（取前10个元素）
        chunk = dialogues_ds[start_index:end_index]  # 切片取出当前份的元素
        result_list.extend(chunk)  # 将当前份的元素添加到结果列表中

    # print(result_list)  # 输出结果列表，长度为90
    return result_list


def gen_transfer_ds(single_ds, div_ds):
    for ds in div_ds:
        # act_prompt = meta_prompt.format(character)
        # ds['turn_content'][0].update({'prompt': meta_prompt})
        turn_num = ds["max_turns"]  # 对话轮数
        name = ds["character"]
        # 1. 随机生成替换的轮数，0 < 轮数 < turn_num
        replace_turns = random.randint(1, turn_num - 1)  # min:1 max:4

        # 2. 随机生成长度为替换轮数的索引列表
        replace_indices = random.sample(range(1, turn_num), replace_turns)  # 除了第一轮
        swap_dlg_list = get_swap_turn(single_ds, replace_turns, name)  # 用其他role的对话替换
        dlg = ds['content']
        for id, idx in enumerate(replace_indices):
            assert idx != 0
            dlg[2 * idx], dlg[2 * idx + 1] = swap_dlg_list[id]
            dlg[2 * idx]['turn_id'] = idx
            dlg[2 * idx + 1]['turn_id'] = idx
        # for index in range(0, len(dlg), 2):  # step 设为2
        #     if index in replace_indices:
        #
        #         pass
        #     else:
        #         # 不执行替换操作，保持原样
        #         dlg[index + 1]['turn_role'] = ds['character']


dialogues_ds = get_multi_dialogues()

single_ds = get_single_ds(dialogues_ds)
div_ds = divide_ds(dialogues_ds)
gen_transfer_ds(single_ds, div_ds)
print(len(div_ds))
save_path = "../data/seed_data/trans_dialogues.jsonl"
# os.makedirs(save_path, exist_ok=True)

with open(save_path, "w") as file:
    for item in div_ds:
        # 将每个 JSON 对象写入文件的一行
        json.dump(item, file)
        file.write('\n')

