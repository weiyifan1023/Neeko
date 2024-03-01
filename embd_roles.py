import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModel
)

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

def read_profile(path):
    with open(path, 'r', encoding='utf-8') as fp:
        text = fp.read().strip()
    parts = text.split('\n\n')
    assert parts[0].startswith('# '), parts[0]
    agent_profile = []
    for p in parts[1:]:
        agent_profile.append(p.strip())
    return agent_profile[0]

def main(
    encoder_path: str = '/path/to/your/deberta-v3-large',
    seed_data_path: str = '/path/to/your/seed_data',
    save_path: str = '/path/to/save/your/role_embds'
    ):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for k in ROLE_PROFILE_MAPPING.keys():
        ROLE_PROFILE_MAPPING[k] = read_profile(os.path.join(seed_data_path, "profiles/wiki_" + k + ".txt"))

    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    encoder = AutoModel.from_pretrained(encoder_path)

    for k,v in ROLE_PROFILE_MAPPING.items():
        input = tokenizer.encode(v, return_tensors="pt")
        out = encoder(input)
        cls_token = out.last_hidden_state[:, 0, :].reshape(-1)
        torch.save(cls_token, os.path.join(save_path, k + ".pth"))


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)