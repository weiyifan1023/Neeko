from transformers import AutoModelForCausalLM, GenerationConfig, LlamaTokenizer, AutoTokenizer, AutoModel
from typing import Union, List
import json
import torch
from tqdm import tqdm
from moelora import PeftModel
import argparse
import os
import csv

ROLE_PROFILE_MAPPING={
        "Beethoven": "",
        "Caesar": "",
        "Cleopatra": "",
        "Hermione": "",
        "Martin": "",
        "Newton": "",
        "Socrates": "",
        "Spartacus": "",
        "Voldemort": "",
    }
for k in ROLE_PROFILE_MAPPING.keys():
    ROLE_PROFILE_MAPPING[k] = torch.load(os.path.join("/path/to/your/role_embds", k + ".pth")).unsqueeze(0).cuda()


def read_profile(path):
    with open(path, 'r', encoding='utf-8') as fp:
        text = fp.read().strip()
    parts = text.split('\n\n')
    assert parts[0].startswith('# '), parts[0]
    agent_profile = []
    for p in parts[1:]:
        agent_profile.append(p.strip())
    return agent_profile[0]

ROLE_PROFILE_TEXT={}
for k in ROLE_PROFILE_MAPPING.keys():
    profile = read_profile(os.path.join("/path/to/your/seed_data/profiles", "wiki_" + k + ".txt"))
    ROLE_PROFILE_TEXT[profile] = k

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_s_bert():
    tokenizer = AutoTokenizer.from_pretrained('KBLab/sentence-bert-swedish-cased')
    model = AutoModel.from_pretrained('KBLab/sentence-bert-swedish-cased')
    return tokenizer, model

def parse_arguments():
    parser = argparse.ArgumentParser(description="Infer")

    parser.add_argument(
        "--infer_path", type=str, default="/home/tongxuluo/Neeko/seed_data/questions/generated_agent_interview_Beethoven.json", help="path of json."
    )
    parser.add_argument(
        "--save_path", type=str, default='/home/tongxuluo/Neeko/results/Neeko/Beethoven_single.json'
    )
    parser.add_argument(
        "--LLM", type=str, default="/home/tongxuluo/models/Llama-2-7b-hf"
    )
    parser.add_argument(
        "--character", type=str, default="Beethoven"
    )
    parser.add_argument(
        "--lora_path", type=str, default="/home/tongxuluo/Neeko/ckpt/neeko/wo_caesar/20240203140411"
    )
    parser.add_argument(
        "--resume_id", type=int, default=0
    )
    parser.add_argument(
        '--multi-turns', action='store_true', help='Enable multi-turns mode'
    )
    args = parser.parse_args()

    return args


def generate_prompt(character: str, inputs: List):
    prompt = """I want you to act like {character}. I want you to respond and answer like {character}, using the tone, manner and vocabulary {character} would use. You must know all of the knowledge of {character}. 

The status of you is as follows:
Location: Coffee Shop - Afternoon
Status: {character} is casually chatting with a man from the 21st century. {character} fully trusts the man who engage in conversation and shares everything {character} knows without reservation.

The interactions are as follows:

{history}{character} (speaking): """
    history = ""
    for dialog in inputs:
        history += f"{dialog['role']} {dialog['action']}: {dialog['content']}" + "</s>"
    prompted = prompt.format(character=character, history=history)
    return prompted

def evaluate(
            tokenizer,
            model,
            character,
            inputs=None,
            temperature=0.1,
            top_p=0.7,
            top_k=40,
            num_beams=3,
            max_new_tokens=512,
            **kwargs,
    ):
        prompt = generate_prompt(character, inputs)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        print(output)
        return output.split(f"(speaking): ")[-1].strip().replace("</s>", "")


def main(args):
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    tokenizer = LlamaTokenizer.from_pretrained(args.LLM, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
            args.LLM,
            # load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ) # fix zwq
    model = PeftModel.from_pretrained(
            model,
            args.lora_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    if hasattr(model, "global_role_embd"):
        s_tokenizer, s_bert = get_s_bert()
        scores = []
        for k,v in ROLE_PROFILE_TEXT.items():
            sentences = [f'I want you to act like {args.character}', k]
            encoded_input = s_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

            with torch.no_grad():
                model_output = s_bert(**encoded_input)

            # Perform pooling. In this case, max pooling.
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings1 = sentence_embeddings[0]
            embeddings2 = sentence_embeddings[1]
            embeddings1 /= embeddings1.norm(dim=-1, keepdim=True)
            embeddings2 /= embeddings2.norm(dim=-1, keepdim=True)

            cosine_scores = embeddings1 @ embeddings2.t()
            scores.append(cosine_scores.item())
        max_score = max(scores)
        index = scores.index(max_score)
        embd_key = list(ROLE_PROFILE_MAPPING.keys())[index]
            
        model.global_role_embd.append(ROLE_PROFILE_MAPPING[embd_key])

    with open(args.infer_path, 'r') as file:
        test_set = []
        if args.multi_turns:
            for line in file:
                json_obj = json.loads(line)
                test_set.append(json_obj)
        else:
            test_set = json.load(file)
    for i, one in enumerate(tqdm(test_set)):
        if i < args.resume_id - 1:
            continue
        if args.multi_turns:
            pass
            inputs = []
            for j in range(one["max_turns"]):
                inputs.append({
                    "role": one["content"][2 * j]["turn_content"][0]["role"],
                    "action": one["content"][2 * j]["turn_content"][0]["action"],
                    "content": one["content"][2 * j]["turn_content"][0]["content"],
                })
                res = evaluate(tokenizer=tokenizer, model=model, character=args.character, inputs=inputs)
                one["content"][2 * j + 1]["turn_content"][0]["content"] = res
                inputs.append({
                    "role": one["content"][2 * j + 1]["turn_content"][0]["role"],
                    "action": one["content"][2 * j + 1]["turn_content"][0]["action"],
                    "content": one["content"][2 * j + 1]["turn_content"][0]["content"],
                })
            if not os.path.exists(args.save_path):
                with open(args.save_path, 'w') as file:
                    pass
            with open(args.save_path, 'a') as file:
                json.dump(one, file)
                file.write('\n')
        else:
            outline = {
                "topic_id": one["topic_id"],
                "question": one["question"],
            }
            inputs=[{
                "role": "Man",
                "action": "(speaking)",
                "content": one["question"]
            }]
            res = evaluate(tokenizer=tokenizer, model=model, character=args.character, inputs=inputs)
            reply = {
                "role": args.character,
                "action": "(speaking)",
                "content": res,
            }
            outline["reply"] = reply
            if not os.path.isfile(args.save_path):
                with open(args.save_path, 'w') as file:
                    json.dump([], file)
            with open(args.save_path, 'r+') as file:
                file_data = json.load(file)
                file_data.append(outline)
                file.seek(0)
                json.dump(file_data, file, indent=4)

if __name__ == "__main__":
    args = parse_arguments()
    main(args=args)
