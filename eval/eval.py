import json
import argparse
import re
# 将字符串存入json文件
def save_data_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# 从json文件读取字符串
def read_data_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def read_data_from_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data

def save_data_to_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')

# @title Parsing JSON output
def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output


def extract_think_content(text: str):
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def extract_answer_content(text: str):
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def eval(file_path):

    opt = ['A', 'B', 'C', 'D', 'E']
    data = read_data_from_jsonl(file_path)
    sum = 0.0
    num = 0
    er = []
    for result in data:
        try:
            # print(parse_json(result["predict"]))
            # print(parse_json(result["label"]))
            num = num + 1
            if 'score' in result.keys():
                sum += result['score']
                continue
            if result["predict"] == result["label"]:
                sum += 1
                continue
            predict = result["predict"]
            label = result["label"]
            predict_ans = extract_answer_content(predict)[0]
            label_ans = extract_answer_content(label)[0]
            if label_ans not in opt:
                num -= 1
                continue
            if predict_ans == label_ans:
                sum += 1
            else:
                er.append(result)

            # print("-----------------------------------------------------")
            # print(result["prompt"])
            # print(f"predict: {predict_ans} | label: {label_ans}")
            # print("-----------------------------------------------------\n\n\n")
        except:
            er.append(result)
            # print("-----------------------------------------------------")
            # print(result["predict"])
            # print("-------------------------------- ---------------------\n\n\n")
            continue
    # save_data_to_json(er, "error_mcq.json")
    # print(f"accuracy: {num/tot*100:.2f}%")
    with open(args.file_path.split('.')[0] + ".txt", "w") as f:
        f.write(f"Scores: {sum}\n")
        f.write(f"Total number of questions: {num}\n")
        f.write(f"Accuracy: {sum / num:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, default='gemini-2.0-flash-thinking-exp_Reasoning_test_KMVE.jsonl', help='eval jsonl file')

    args = parser.parse_args()

    eval(args.file_path)
