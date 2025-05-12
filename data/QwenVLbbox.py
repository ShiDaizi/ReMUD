import json
import random
import io
import ast
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import xml.etree.ElementTree as ET
import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
import json



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


def decode_xml_points(text):
    try:
        root = ET.fromstring(text)
        num_points = (len(root.attrib) - 1) // 2
        points = []
        for i in range(num_points):
            x = root.attrib.get(f'x{i+1}')
            y = root.attrib.get(f'y{i+1}')
            points.append([x, y])
        alt = root.attrib.get('alt')
        phrase = root.text.strip() if root.text else None
        return {
            "points": points,
            "alt": alt,
            "phrase": phrase
        }
    except Exception as e:
        print(e)
        return None

def plot_bounding_boxes(im, bounding_boxes, input_width, input_height):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im.copy()
    width, height = img.size
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)

    font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

    try:
      json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
      end_idx = bounding_boxes.rfind('"}') + len('"}')
      truncated_text = bounding_boxes[:end_idx] + "]"
      json_output = ast.literal_eval(truncated_text)

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json_output):
      # Select a color from the list
      color = colors[i % len(colors)]

      # Convert normalized coordinates to absolute coordinates
      abs_y1 = int(bounding_box["bbox_2d"][1]/input_height * height)
      abs_x1 = int(bounding_box["bbox_2d"][0]/input_width * width)
      abs_y2 = int(bounding_box["bbox_2d"][3]/input_height * height)
      abs_x2 = int(bounding_box["bbox_2d"][2]/input_width * width)

      if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

      if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

      # Draw the bounding box
      draw.rectangle(
          ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
      )

      # Draw the text
      if "label" in bounding_box:
        draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    # Display the image
    # img.show()
    return img



def inference(img_url: str, prompt: str, device: str, system_prompt="You are a helpful assistant", max_new_tokens=2048, **kwargs):
  image = Image.open(img_url)
  messages = [
    {
      "role": "system",
      "content": system_prompt
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": prompt
        },
        {
          "image": img_url
        }
      ]
    }
  ]
  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  print("input:\n",text)
  inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)

  output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
  generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
  output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
  print("output:\n",output_text[0])

  input_height = inputs['image_grid_thw'][0][1]*14
  input_width = inputs['image_grid_thw'][0][2]*14
  
  return output_text[0], input_height, input_width


def qwenvl_bbox(image_path: str, device: str, system_prompt: str, prompt: str, **kwargs):

    ## Use a local HuggingFace model to inference.
    response, input_height, input_width = inference(img_url=image_path, prompt=prompt, system_prompt=system_prompt, device=device)

    image = Image.open(image_path)
    # image.thumbnail([640,640], Image.Resampling.LANCZOS)
    image = image.resize((input_width, input_height))
    img = plot_bounding_boxes(image, response, input_width, input_height)
    return image, img, response

    ## Use an API-based approach to inference. Apply API key here: https://bailian.console.alibabacloud.com/?apiKey=1
    # from qwen_vl_utils import smart_resize
    # os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here' 
    # min_pixels = 512*28*28
    # max_pixels = 2048*28*28
    # image = Image.open(image_path)
    # width, height = image.size
    # input_height,input_width = smart_resize(height,width,min_pixels=min_pixels, max_pixels=max_pixels)
    # response = inference_with_api(image_path, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
    # plot_bounding_boxes(image, response, input_width, input_height)


def save_data_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_data_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def crop_images_by_bbox(image, box_image, bbox_list, page, save_folder, **kwargs):
    """
    根据传入的图片路径和 bbox 列表，截取对应区域的图片并保存
    :param image_path: 图片对象
    :param bbox_list: 包含多个 bbox 的列表，每个 bbox 是一个包含四个元素的元组 (left, upper, right, lower)
    """
    if not bbox_list:
        print("bbox_list error")
        return

    if not os.path.exists(f'{save_folder}'):
        os.makedirs(f'{save_folder}')

    if not os.path.exists(f'{save_folder}/{page}'):
        os.makedirs(f'{save_folder}/{page}')

    box_image.save(f'{save_folder}/{page}/page_{page}.jpg')
    box_list = []
    try:
        for i, item in enumerate(bbox_list):
            # 确保 bbox 格式正确

            bbox = item['bbox_2d']
            if len(bbox) != 4:
                print(f"第 {i + 1} 个 bbox 格式不正确，应为 (left, upper, right, lower)，跳过该 bbox。")
                continue

            left, upper, right, lower = bbox

            # 检查坐标是否合法
            if left < 0 or upper < 0 or right > image.width or lower > image.height or left >= right or upper >= lower:
                print(f"第 {i + 1} 个 bbox 坐标不合法，跳过该 bbox。")
                continue

            # 截取图片
            cropped_image = image.crop(bbox)

            # 生成保存文件名
            save_path = f"{save_folder}/{page}/page_{page}{i + 1}.jpg"
            item['images'] = save_path
            box_list.append(item)
            # 保存截取后的图片
            cropped_image.save(save_path)
            print(f"第 {i + 1} 个截取图片已保存为 {save_path}")

    except FileNotFoundError:
        print(f"未找到图片文件: {page}")
    except Exception as e:
        print(f"发生错误: {e}")
    
    save_data_to_json(box_list, f"{save_folder}/{page}/page_{page}.json")


def page_process(page: int, image_folder: str, **kwargs):
    image_path = f'{image_folder}/page_{page}.jpg'
    if not os.path.exists(image_path):
        print(f"page_{page} not exists.\n")
        return
    
    print(f"Processing page_{page}...\n")
    
    image, box_image, response = qwenvl_bbox(image_path=image_path, **kwargs)
    try: 
        bbox_list = json.loads(parse_json(response))
    except:
        print(f"{page} json error")
        return

    crop_images_by_bbox(image, box_image, bbox_list, page=page, **kwargs)

    print(f"Finish page_{page}.\n")

def page2json(start_page: int, end_page: int, **kwargs):
    for page in range(start_page, end_page+1):
        page_process(page, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Qwen2.5-VL-7B bounding box.")
    parser.add_argument('-s', '--start_page', type=int, default=123)
    parser.add_argument('-e', '--end_page', type=int, default=124)
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-m', '--model', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct', help='Qwen2.5-VL model path.')
    parser.add_argument('--image_folder', type=str, default='./image')
    parser.add_argument('--save_folder', type=str, default='./save_bbox')
    parser.add_argument(
       '--system_prompt',
        type=str,
        default="""你会得到一页超声pdf书籍的图像，其中有会有超声图像作为插图，请你框出图中的超声图像，以json格式输出其bbox坐标：
        - 请你提取每张图片的标题图例信息，输出在json的'caption'关键字中，，具体输出格式如下：
        ```json
        [
        {"bbox_2d": [765, 394, 891, 550], "label": "超声图像1", "caption": "图片标题，编号，图注信息等"},
        {"bbox_2d": [132, 234, 234, 350], "label": "超声图像2", "caption": "图片标题，编号，图注信息等"}
        ]
        ```
        - 如果没有超声图像，则输出空列表：
        ```json
        []
        ```
        """
    )
    parser.add_argument(
       '--prompt',
        type=str,
        default="框出图中的超声图像，以json格式输出其bbox坐标和相关文字"
    )

    args = parser.parse_args()

    additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]


    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained(args.model)

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)


    page2json(
       start_page=args.start_page,
       end_page=args.end_page,
       image_folder=args.image_folder,
       save_folder=args.save_folder,
       device=args.device,
       system_prompt=args.system_prompt,
       prompt=args.prompt
    )


