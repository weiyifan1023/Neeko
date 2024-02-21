import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


def find_rankings(list_a, list_b):
    rankings = {}  # 用于存储列表B中字符串的排名

    # 构建字典，将列表B中字符串的子串映射到它们的索引位置
    for i, item in enumerate(list_b):
        for substring in list_a:
            if substring in item:
                rankings[substring] = i + 1  # 索引从1开始
                break  # 只需找到一个匹配即可

    # 返回列表A中字符串在列表B中的排名，如果没有出现则为0
    result = [rankings.get(item, 0) for item in list_a]

    return result


def beam_search(model, tokenizer, input_ids, top_size=20, prompt_len=8, max_tgt_len=3):
    # 生成候选词

    # 使用beam生成第一个token，20组(候选实体)
    beam_output = model.generate(
        input_ids,
        max_length=prompt_len + 1,
        num_beams=top_size,
        early_stopping=True,
        num_return_sequences=top_size,
        output_scores=True,
    )

    # 使用greed模式 对20组的候选生成后面的token，最大长度为候选实体的最大长度
    greed_output = model.generate(
        beam_output,
        max_length=prompt_len + max_tgt_len,
        num_beams=1,
        early_stopping=True,
        num_return_sequences=1,
        output_scores=True,
    )
    print(greed_output)

    # Decode and return the top beam along with scores
    beam = []
    for i in range(top_size):
        beam.append(tokenizer.decode(beam_output[i], skip_special_tokens=True))

    greed = []
    for i in range(top_size):
        greed.append(tokenizer.decode(greed_output[i], skip_special_tokens=True))

    pred = []
    for i in range(top_size):
        pred.append(tokenizer.decode(greed_output[i][prompt_len:], skip_special_tokens=True))

    pred_id = []

    return beam, greed, pred


# print("Input: ", input_text)
#
# for i in beam:
#     print(i)
#
# print("***greed*****")
# for i in greed:
#     print(i)
#
# print("********")
# for i in pred:
#     print(i)
#
# # 排名
# out = find_rankings(target_new, pred)
# print("ranking result:", out)
#
# print(max_tgt_len)
# print([len(tokenizer.encode(i)) for i in target_new])


# return a list of relevance for supplied list,
def get_relevances(pred_list, update_list, ori_list):
    edit_set = list(set(update_list) - set(ori_list))  # 差集
    keep_set = list(set(update_list) & set(ori_list))  # 交集
    rel = []
    for i, pred in enumerate(pred_list):
        obj = update_list[i]  # gold
        r = 0
        if obj == pred:
            if obj in edit_set:
                r = 2
            if obj in keep_set:
                r = 1
        else:
            if obj in edit_set:
                r = -2
            if obj in keep_set:
                r = -1
        rel.append(r)

    return rel


# def normalize_scores(scores):
#     """
#     使用min-max归一化方法将得分归一化到0到1之间
#     :param scores: 一个包含得分的列表
#     :return: 归一化后的得分列表
#     """
#     min_score = min(scores)
#     max_score = max(scores)
#     if max_score == min_score:
#         return [0.5] * len(scores)
#     return [(s - min_score) / (max_score - min_score) for s in scores]


def ndcg_at_k(ideal_scores, pred_scores, k):
    """
    计算nDCG@k评价指标
    :param true_scores: 一个列表，包含真实相关度得分（例如[1, 2, 3, 0, 2]）长度为N
    :param pred_scores: 一个列表，包含推荐结果的相关度得分（例如[3, 1, 2, 0, 2]）长度为N
    :param k: 要计算nDCG的排名长度
    :return: nDCG@k和排序后的预测相关度得分
    """
    # 将真实相关度得分和预测相关度得分归一化到0到1之间
    # true_scores_norm = normalize_scores(true_scores)
    # pred_scores_norm = normalize_scores(pred_scores)

    # 计算DCG@k
    dcg_k = 0
    for i in range(k):
        dcg_k += (2 ** pred_scores[i]) / np.log2(i + 2)
        # 注意：这里使用i+2而不是i+1是因为i从0开始，排名从1开始

    # 计算IDCG@k
    # ideal_scores = np.sort(true_scores_norm)[::-1] # 排序真实相关度得分
    idcg_k = 0
    for i in range(k):
        idcg_k += (2 ** ideal_scores[i]) / np.log2(i + 2)  # a^rel / log_2(i+1)

    # 如果IDCG为0，返回0,避免除以0错误
    if idcg_k == 0:
        return 0

    # 计算nDCG@k
    ndcg_k = dcg_k / idcg_k

    # 将预测相关度得分排序，返回排序后的前k个值
    # pred_scores_sorted = np.sort(pred_scores_norm)[::-1][:k]

    return ndcg_k  # , pred_scores_sorted


def ecg(ideal_scores, pred_scores):
    ecg_score = 0
    for i in range(len(pred_scores)):
        ecg_score += ndcg_at_k(ideal_scores, pred_scores, i + 1)
        print(ecg_score / (i + 1))  # 取平均
    return ecg_score


if __name__ == "__main__":
    # true_scores = [2, 1.6, 1.8, 1.7, 1.75, 1.78, 1.7, 1.6, 1.7, 1.5, 1.5, 1.48, 1.43, 1.43, 1.5, 1.5, 1.2, 1, 0.9]
    # pred_scores = [1.51, 1, 1.36, 1.17, 1.16, 1.17, 1.166, 1.55, 1.161, 1.01, 1.05, 0.96, 0.94, 0.95, 1.04, 1.67, 1.27,
    #                1.21, 1.04]
    #
    # target_true = [
    #     "Prix Renaudot",
    #     "Order of the October Revolution",
    #     "Order of Friendship of Peoples",
    #     "Lenin Peace Prize",
    #     "Croix de guerre 1939–1945",
    #     "Croix de guerre 1914–1918",
    #     "Médaille militaire"
    # ]
    #
    # target_new = [
    #     "Prix Renaudot",
    #     "Order of the October Revolution",
    #     "Academy Award for Best Costume Design, Black-and-White",
    #     "Mårbacka Award",
    #     "Knight of the Order of the Dannebrog",
    #     "Croix de guerre 1914–1918",
    #     "Royal Family Order of King Harald V of Norway"
    # ]
    #
    # target_pred = [
    #     "Prix ",
    #     "Order of the October Revolution",
    #     "Academy Award for Best Costume Design, Black-and-White",
    #     "Mårbacka Award",
    #     "Croix de guerre 1939–1945",
    #     "Croix de guerre 1914–1918",
    #     "Médaille militaire"
    # ]

    # Generation: Inference Stage
    model_name = "/data/suyisong/checkpoint/gpt2-xl"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_text = "The Byzantine Empire spanned across multiple continents, including"
    target_new = ["Europe",
                  "Africa",
                  "North America"]

    target_true = ["Europe",
                   "Africa",
                   "Asia"]

    # input_text = "The leaders of the People's Republic of China include"
    # target_new = ['Xi Jinping', 'Jiang Zemin', 'Hu Jintao']

    max_tgt_len = max([len(tokenizer.encode(i)) for i in target_new])

    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    _, prompt_len = input_ids.shape

    beam, greed, pred = beam_search(model, tokenizer, input_ids, prompt_len=prompt_len, max_tgt_len=max_tgt_len)

    target_pred = pred[:len(target_new)]
    pred_scores = get_relevances(target_pred, target_new, target_true)
    true_scores = get_relevances(target_new, target_new, target_true)
    # result = ndcg_at_k(true_scores, pred_scores, len(true_scores))

    print("total score:", ecg(true_scores, pred_scores))
