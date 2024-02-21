from rouge import Rouge

a = ["i am a student from xx school"]  # 预测摘要 （可以是列表也可以是句子）
b = ["i am a student from school on china"]  # 真实摘要

rouge = Rouge(["rouge-l"])
rouge_score = rouge.get_scores(a, b)
# print(rouge_score[0]["rouge-1"])
# print(rouge_score[0]["rouge-2"])
print(rouge_score[0]["rouge-l"])
