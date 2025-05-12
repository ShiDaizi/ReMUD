import json
import os
import copy
import base64
import random
from openai import OpenAI # type: ignore
import shutil
import argparse

# 将字符串存入json文件
def save_data_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# 从json文件读取字符串
def read_data_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# @title Parsing JSON output from Qwen2.5-VL cookbook
def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def get_jpg_filenames_without_ext(folder_path):
    file_names = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jpg'):
                name_without_ext = os.path.splitext(file)[0]
                file_names.append(name_without_ext)
    return file_names


def has_jpg_files(folder_path):
    file_list = os.listdir(folder_path)
    for file in file_list:
        if file.endswith('.jpg'):
            return True
    return False


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_instruction(image_path: str, prompt: str, system_prompt: str, api_model: str, api_url: str, api_key=str, **kwargs):
    print(f"model: {api_model} | image_path: {image_path}")
    client = OpenAI(
        base_url= api_url,
        # sk-xxx替换为自己的key
        api_key= api_key
    )      
    base64_image = encode_image(image_path)

    messages = []

    messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {
                    "tuype": "text",
                    "text": prompt
                }
            ]})
    
    completion = client.chat.completions.create(
        model=api_model,
        messages=messages
    )

    return completion


def check(conversation):
    # print(conversation)
    for index in range(len(conversation)):
        # print(index, conversation[index]["from"])
        if index % 2 == 0 and conversation[index]["from"] != "human":
            return True
        if index % 2 == 1 and conversation[index]["from"] != "gpt":
            return True
    
    return False



def page_process(page: int, bbox_folder, **kwargs):
    image_path = f'{bbox_folder}/{page}/page_{page}.jpg'
    json_path = f'{bbox_folder}/{page}/page_{page}.json'
    prompt = json.dumps(read_data_from_json(json_path))

    try:
        completion = get_instruction(image_path=image_path, prompt=prompt, **kwargs)
        content = completion.choices[0].message.content
        print(content)
        data = json.loads(parse_json(content))
    except:  
        print(f'page_{page} gpt request error.')
        return []
        
    
    return data


def page2json(start_page: int, end_page: int, bbox_folder: str, **kwargs):
    for page in range(start_page, end_page+1):
        if not os.path.exists(f"{bbox_folder}/{page}"):
            continue
        
        print(f'Processing page_{page}...\n')
        data = page_process(page, bbox_folder, **kwargs)
        save_data_to_json(data, file_path=f'{bbox_folder}/{page}/gpt_page_{page}.json')
        print(f'Finish page_{page}.\n')


def check_and_clean_sharegpt(json_list):
    """
    检查 JSON 列表中的每个元素是否符合 ShareGPT 格式，
    若不符合则删除该元素，最后返回清理后的列表
    :param json_list: 待检查的 JSON 列表
    :return: 清理后的 JSON 列表
    """
    cleaned_list = []
    for item in json_list:
        if isinstance(item, dict) and 'conversations' in item:
            conversations = item['conversations']
            valid = True
            # 检查对话是否以 'human' 开头
            if conversations and conversations[1].get('from') != 'human':
                valid = False
            for i in range(2, len(conversations)):
                prev_from = conversations[i - 1].get('from')
                current_from = conversations[i].get('from')
                # 检查是否交替出现
                if (prev_from == 'human' and current_from != 'gpt') or (prev_from == 'gpt' and current_from != 'human'):
                    valid = False
                    break
            if valid:
                cleaned_list.append(item)
    return cleaned_list

def copy_file_to_folder(file_address: str, target_folder: str, new_name: str):
    """
    将指定文件复制到目标文件夹中，可选择为复制后的文件指定新名称
    :param file_address: 源文件的地址
    :param target_folder: 目标文件夹的地址
    :param new_name: 复制后文件的新名称，若为 None 则使用原文件名
    :return: 若复制成功返回 True，若出现异常返回 False
    """
    try:
        # 检查目标文件夹是否存在，如果不存在则创建
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # 确定目标文件名
        if new_name is None:
            file_name = os.path.basename(file_address)
        else:
            file_name = new_name

        # 拼接目标文件的完整路径
        target_file = os.path.join(target_folder, file_name)

        # 复制文件
        shutil.copy2(file_address, target_file)
        print(f"文件 {file_address} 已成功复制到 {target_folder}，新文件名为 {file_name}")
        return True
    except Exception as e:
        print(f"复制文件时出现错误: {e}")
        return False



def merge_page_json(page: int, folder: str, bbox_folder:str, prefix: str, copy_file: str, **kwargs):
    gpt_json = read_data_from_json(f"{bbox_folder}/{page}/gpt_page_{page}.json")
    bbox_json = read_data_from_json(f"{bbox_folder}/{page}/page_{page}.json")
    if len(gpt_json) != len(bbox_json):
        return []
    for index, chat in enumerate(gpt_json):
        image_name = os.path.basename(bbox_json[index]['images'])
        if copy_file:
            copy_file_to_folder(file_address=f"{bbox_folder}/{page}/{image_name}", target_folder=folder, new_name=f"{prefix}{image_name[4:]}")
        
        chat['images'] = [f"./{folder}/{prefix}{image_name[4:]}"]
        chat['conversations'][0]["value"] += "\n<image>"
    
    return gpt_json

def merge_page(start_page: int, end_page: int, bbox_folder:str, **kwargs):
    merge_list = []
    for page in range(start_page, end_page+1):
        if not os.path.exists(f"{bbox_folder}/{page}"):
            continue
        merge_list.extend(merge_page_json(page=page, bbox_folder=bbox_folder, **kwargs))
    return merge_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQA generation.")
    parser.add_argument('-s', '--start_page', type=int, default=124)
    parser.add_argument('-e', '--end_page', type=int, default=124)
    parser.add_argument('--api_model', type=str, default='gpt-4o')
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--api_url', type=str, default='')
    parser.add_argument('--bbox_folder', type=str, default='./save_bbox')
    parser.add_argument('--prefix', type=str, default='demo')
    parser.add_argument('--copy_file', type=lambda x: (str(x).lower() == 'true'), default=True, help='Set to true or false')
    parser.add_argument('--vqa_prompt', type=str, default='是一位医学超声助手，请回答相关问题。')
    parser.add_argument('--system_prompt', type=str, default="""
    你是一名超声医学人工智能助手。你收到了超声医学书籍某一页的图片和一个json列表，json列表与图中标注的超声图像对应，请你完成以任务：
     - 根据json列表每一个元素代表一张超声图，根据json里的caption关键字和这一页与这张超声图对应的文字（如标题，提及的段落），生成一系列对应这张图的问答对话，如“这是什么超声图片？”“图中能看到什么特征？”等，以sharegpt的形式给出，回答中不出现图片编号，序号等信息
     - 将输出标准化为的json格式，其具体为sharegpt格式，以供sft微调，即"```json\n[{"conversations": [{"from": "human", "value": "问题"}, {"from": "gpt", "value": "答案"}]}， {"conversations": [{"from": "human", "value": "问题"}, {"from": "gpt", "value": "答案"}]}]\n```"
     - 输出的sharegpt格式要与json列表对应，即输出列表的第一个对应给你的json列表的第一个，然后可以包含多轮的对话，满足sharegpt格式即可
     - 回答中不出现图片编号，序号等信息，对话尽量多轮且详细，对话围绕这张超声图片展开，的答案能在书上找到，即根据书上内容生成问答，并且你可以根据你知道的内容再补充相关知识的问答，每张图的对话相互独立不要出现互相提及
     - 要求输出的json列表元素与给你的json列表元素个数相同即一一对应，即每张超声图对应一段多轮sharegpt格式的对话
    """)
    parser.add_argument('--require_api', type=lambda x: (str(x).lower() == 'true'), default=True, help='Set to true or false')
    
    args=parser.parse_args()
    
    if args.require_api is True:
        page2json(
            start_page=args.start_page,
            end_page=args.end_page,
            bbox_folder=args.bbox_folder,
            api_model=args.api_model,
            api_key=args.api_key,
            api_url=args.api_url,
            system_prompt=args.system_prompt
        )

    merge_list = merge_page(args.start_page, args.end_page, bbox_folder=args.bbox_folder, folder=args.prefix+"_images", prefix=args.prefix, copy_file=args.copy_file)
    for chat in merge_list:
        prompt = args.vqa_prompt
        sys_prompt = {"from": "system", "value": prompt}
        chat['conversations'].insert(0, copy.deepcopy(sys_prompt))
    merge_list = check_and_clean_sharegpt(merge_list)
    save_data_to_json(merge_list, f'./{args.prefix}.json')