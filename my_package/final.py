import re
import os
import json
from sacrebleu.metrics import BLEU


def extract_json_from_label_studio(json_path, new_json_path):
    with open(json_path, "r") as f:
        results = json.load(f)
    doub_loop = [f'{class_type}-edit{num}' for class_type in ['question', 'answer', 'solution'] for num in range(1, 8)]
    for data in results:
        for key in doub_loop:
            if (key in data) and isinstance(data[key], dict):
                remove_redundant(data, key)
    with open(new_json_path, 'w', encoding='utf-8') as f:
        data_json = json.dumps(results, ensure_ascii=False,indent=4)
        f.write(data_json)


def remove_redundant(data_dict, key):
    if len(set(data_dict[key]['text'])) == 1:
        data_dict[key] = data_dict[key]['text'][0]
    else:
        try:
            data_dict[key]['text'].remove(data_dict[key.replace("-edit", "")])
            data_dict[key] = data_dict[key]['text'][0]
        except Exception as e:
            print(f"An error occurred dealing id={data_dict['id']}, key={key}: {e}")
            return []


extract_json_from_label_studio("C:/Users/廉/Downloads/project-1-at-2024-07-09-12-56-5ef4f7e4.json", "../1.json")



def extract_answer(annotation_dict, answer):
    pattern = r"答案是(\{.+\})"
    # 将 'None' 替换为 '"None"'，以便 JSON 解析
    answer = answer.replace('None', '"None"')
    match = re.findall(pattern, answer)
    if match:
        try:
            answers = json.loads(match[-1])
        except json.JSONDecodeError:
            answers = {}
        return answers
    else:
        ques_num = sum([1 for i in list(annotation_dict.keys()) if re.match(r'question-edit[0-9]', i)])
        model_answers = {f"A{num}": "" for num in range(1, ques_num + 1)}
        return model_answers


def calculate_bleu_score(predictions, references):
    bleu = BLEU()
    bleu_score = bleu.corpus_score(predictions, references)
    return bleu_score.score


def check_annotation(new_json_path, yi_new_json_path, key='gpt4_answer'):
    # 加载annotations
    with open(new_json_path, "r") as f:
        annotations = json.load(f)

    # 加载gpt4预测
    dict_gpt4_predict = {}
    with open(yi_new_json_path, 'r', encoding='utf-8') as fr:
        data = fr.readlines()
        for line in data:
            line = json.loads(line)
            dict_gpt4_predict[line['image']] = {"gpt4_answer": line["gpt4_answer"]}

    right = 0
    wrong = 0
    all_predictions = []
    all_references = []

    for data in annotations:
        img_name = data["image"].replace(r"/data/local-files/?d=table_use_img/", "")
        if img_name in dict_gpt4_predict:
            gpt4v_answer = dict_gpt4_predict[img_name][key]
            print(f"***********id={data['id']}***********\n{gpt4v_answer}")
            model_answers = extract_answer(data, gpt4v_answer)
            gold_answer = extract_gold_answer(data)

            # 将预测结果和参考结果添加到列表中，以便计算BLEU分数
            all_predictions.append(gpt4v_answer)
            all_references.append([gold_answer])

            # 检查答案是否正确
            for answer_key, model_answer in model_answers.items():
                answer_key = answer_key.replace("A", "")
                answer_key_full = f"answer-edit{answer_key}"
                if answer_key_full in data:
                    answer = str(data[answer_key_full])
                if isinstance(model_answer, float):
                    model_answer = str(model_answer)
                elif not isinstance(model_answer, str):
                    model_answer = ""
                if (answer in model_answer) or model_answer.strip() == answer.strip():
                    right += 1
                    print(f"Q{answer_key} ✔: Model Answer={model_answer}. Expected: {answer}")
                else:
                    wrong += 1
                    print(f"Q{answer_key} ✖️ : Model Answer={model_answer}. Expected: {answer}")

    # 计算整体BLEU分数
    overall_bleu = calculate_bleu_score(all_predictions, all_references)
    print(f"Overall BLEU score: {overall_bleu}")

    print(f"right:{right}, wrong:{wrong}, accuracy={round(right / (right + wrong), 2)}")
# 需要实现的函数，用于从annotations中提取正确的答案


def extract_gold_answer(annotation_dict):
    gold_answer = annotation_dict.get("answer-edit1", "")
    return gold_answer


# 调用check_annotation函数，传入相应的文件路径
check_annotation("../1.json", "2.json")