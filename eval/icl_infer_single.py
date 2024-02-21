from transformers import AutoModelForCausalLM, GenerationConfig, LlamaTokenizer
from typing import Union, List
from eval_utils import read_profile, seed_data_dir, get_api_key, get_character_names
import json
import torch
from tqdm import tqdm
import argparse
import os
from time import sleep
from openai import OpenAI

client = OpenAI(api_key=get_api_key())


def parse_arguments():
    parser = argparse.ArgumentParser(description="Infer")

    parser.add_argument(
        "--infer_path", type=str, default="../data/seed_data/questions/generated_agent_interview_{name}.json",
        help="path of json."
    )
    parser.add_argument(
        "--save_path", type=str,
        default='../data/gen_results/interview_multi/{name}_{baseline}_result/{name}_single.json'
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="/data/suyisong/checkpoint/llama-2-7b-chat"
    )
    parser.add_argument(
        "--gpt_api", type=str, default="gpt-3.5-turbo"
    )
    parser.add_argument(
        "--character", type=str, default="Beethoven"
    )
    parser.add_argument(
        "--lora_path", type=str, default="/home/tongxuluo/Neeko/ckpt/lora/20240128135955"
    )
    parser.add_argument(
        "--baseline", type=str, default="llama-7b-2-chat"
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
    # ICL Prompt
    gpt_prompt = """I want you to act like {character}. I want you to respond and answer like {character}, using the tone, manner and vocabulary {character} would use. You must know all of the knowledge of {character}. Reply must be brief and concise.

The status of you is as follows:
Location: Coffee Shop - Afternoon
Status: {character} is casually chatting with a man from the 21st century. {character} fully trusts the man who engage in conversation and shares everything {character} knows without reservation.

Example output:
Character1 (speaking): Detailed utterance ...

Character2 (speaking): Detailed utterance ...

The interactions are as follows:

{history}{character} (speaking):"""

    history = ""
    for dialog in inputs:
        history += f"{dialog['role']} {dialog['action']}: {dialog['content']}" + "</s>"
    prompted = gpt_prompt.format(character=character, history=history)
    return prompted


def gpt4_evaluator(inputs, character):
    prompt = generate_prompt(character, inputs).replace("</s>", "\n")
    request_num = 0
    got_result = False
    response = ""
    while not got_result:
        try:
            # ChatCompletion
            if args.gpt_api == "gpt-3.5-turbo":
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


def evaluate(
        tokenizer,
        model,
        character,
        inputs=None,
        rag=None,
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
    input_len = input_ids.shape[1]
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
    output = tokenizer.decode(s[input_len:].cpu())

    # s[input_len:]
    # result = tokenizer.batch_decode(s[:, input_len:].cpu(), skip_special_tokens=True)
    print("Output: ", output)
    return output.split("\n\n")[0]
    # return output.split(f"{character} (speaking): ")[-1].strip().replace("</s>", "")


def main(args):
    os.makedirs(os.path.dirname(args.save_path.format(name=args.character, baseline=args.baseline)), exist_ok=True)
    if args.baseline != "chatgpt":
        tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt_path,
            # load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    # fix zwq
    # lora
    # model = PeftModel.from_pretrained(
    #         model,
    #         args.lora_path,
    #         torch_dtype=torch.float16,
    #         device_map="auto"
    #     )
    # prompt="I want you to act like Ludwig van Beethoven. I want you to respond and answer like Ludwig van Beethoven, using the tone, manner and vocabulary Ludwig van Beethoven would use. You must know all of the knowledge of Ludwig van Beethoven. \n\nThe status of you is as follows:\nLocation: Nobility ball\nStatus: The grand ballroom of the Viennese nobility was filled with the finest ladies and gentlemen of the city. The air was filled with music and the clinking of glasses as the guests mingled and danced. In the center of the room, a grand piano was being played by none other than Ludwig van Beethoven. He had been invited to perform by his patrons, who had connections with the nobility. Beethoven had developed a reputation as a virtuoso pianist and composer, and his works were now being published by his friend Nikolaus Simrock. The audience was enraptured by Beethoven's performance, and he was basking in the applause and admiration of his patrons.\n\nThe interactions are as follows:\n\n"
    # input="Beethoven (speaking): "
    # res = evaluate(tokenizer=tokenizer, model=model, prompt=prompt, input=input)
    with open(args.infer_path.format(name=args.character), 'r') as file:
        test_set = json.load(file)
    for i, one in enumerate(tqdm(test_set)):
        if i < args.resume_id - 1:
            continue
        outline = {
            "topic_id": one["topic_id"],
            "question": one["question"],
        }
        if args.multi_turns:
            pass
        else:
            inputs = [{
                "role": "Man",
                "action": "(speaking)",
                "content": one["question"]
            }]

        # llms
        # res = evaluate(tokenizer=tokenizer, model=model, character=args.character, inputs=inputs)
        # chatgpt
        res = gpt4_evaluator(character=args.character, inputs=inputs)
        reply = {
            "role": args.character,
            "action": "(speaking)",
            "content": res,
        }
        outline["reply"] = reply

        if not os.path.isfile(args.save_path.format(name=args.character, baseline=args.baseline)):
            with open(args.save_path.format(name=args.character, baseline=args.baseline), 'w') as file:
                json.dump([], file)
        with open(args.save_path.format(name=args.character, baseline=args.baseline), 'r+') as file:
            file_data = json.load(file)
            file_data.append(outline)
            file.seek(0)
            json.dump(file_data, file, indent=4)


if __name__ == "__main__":
    args = parse_arguments()
    name_list = get_character_names()
    for name in name_list:
        args.character = name
        print("Current Character:", name)
        main(args=args)


