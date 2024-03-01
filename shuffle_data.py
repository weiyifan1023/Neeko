import os
import glob
import json
import random


def main(
    data_dir: str = '/path/to/your/character-llm-data/',
    out_path: str = '/path/to/your/character-llm-data/prompted/shuffle.jsonl'
    ):
    
    jsonl_files = glob.glob(os.path.join(data_dir, 'prompted/*.jsonl'))
    data = []

    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r') as file:
            role = jsonl_file.split("prompted_agent_dialogue_")[-1].replace('.jsonl', '')
            for line in file:
                one = json.loads(line)
                one["role"] = role
                one["eot"] = "<|eot|>"
                data.append(one)

    random.shuffle(data)

    with open(out_path, 'w') as jsonl_file:
        for item in data:
            json_line = json.dumps(item)
            jsonl_file.write(json_line + '\n')


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
