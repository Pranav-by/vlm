from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


import json
import os
import torch
import argparse
import time
import traceback
import sys
import io


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    if os.path.exists(args.output_json_path):
        with open(args.output_json_path, 'r') as f:
            results = json.load(f)
        completed_ids = {item['image_id'] for item in results} 
    else:
        directory = os.path.dirname(args.output_json_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        results = []
        completed_ids = set()

    count = 0
    data = load_json(args.input_json_path)
    errors = []
    start_time = time.time()
    for item in data:
        # start_time = time.time()
        if item['id'] in completed_ids:
            continue
        try:           
            image_path = os.path.join(args.image_folder, item['image'])
            image = load_image(image_path)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
            conversation_result = {
                "image_id": item["id"],
            }
            for temperature in args.temperatures:
                conversation_result[f"conversations_{temperature}"] = []
                for conversation in item["conversations"]:
                    if conversation["from"] == "human":
                        inp = conversation["value"].replace("\n<image>", "").replace("<image>\n", "")
                        conversation_result[f"conversations_{temperature}"].append({"from": "human", "value": inp})

                        repeat_count = 0
                        for _ in range(args.repeat): 
                            conv = conv_templates[args.conv_mode].copy()
                            if "mpt" in model_name.lower():
                                roles = ('user', 'assistant')
                            else:
                                roles = conv.roles

                            if image is not None:
                                if model.config.mm_use_im_start_end:
                                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                                else:
                                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                                conv.append_message(conv.roles[0], inp)
                                image = None
                            else:
                                conv.append_message(conv.roles[0], inp)

                            conv.append_message(conv.roles[1], None)
                            prompt = conv.get_prompt()
                            
                            original_stdout = sys.stdout 
                            buffer = io.StringIO() 
                            sys.stdout = buffer 

                            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                            keywords = [stop_str]
                            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                            keywords = [stop_str]
                            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

                            with torch.inference_mode():
                                output_ids = model.generate(
                                    input_ids,
                                    images=image_tensor,
                                    do_sample=True,
                                    temperature=temperature,
                                    max_new_tokens=1024,
                                    streamer=streamer,
                                    use_cache=True,
                                    stopping_criteria=[stopping_criteria])

                            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
                            conv.messages[-1][-1] = outputs

                            buffer.truncate(0) 
                            buffer.seek(0) 
                            sys.stdout = original_stdout

                            repeat_count += 1
                            conversation_result[f"conversations_{temperature}"].append({"from": f"vlm_{repeat_count}", "value": outputs})
                    elif conversation["from"] == "gpt":
                        conversation_result[f"conversations_{temperature}"].append({"from": "ground truth", "value": conversation["value"]})
            results.append(conversation_result)
        except Exception as e:
            error_msg = f"Error processing image {item['image']}: {str(e)}"
            print(error_msg)
            errors.append({'image_id': item['image'], 'error': error_msg, 'traceback': traceback.format_exc()})
        finally:
            elapsed_time = time.time() - start_time
        count += 1
        if count % 100 == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print(f"Finish conversation on {count} samples, process of last 100 samples takes {elapsed_time} seconds.")
            with open(args.output_json_path, 'w') as f:
                json.dump(results, f, indent=4)
            if errors:
                error_file = args.output_json_path.replace('.json', '_errors.json')
                with open(error_file, 'w') as f:
                    json.dump({"errors": errors}, f, indent=4)
    with open(args.output_json_path, 'w') as f:
                json.dump(results, f, indent=4)
    print(f"Results are saved in {args.output_json_path}.")
    if errors:
        error_file = args.output_json_path.replace('.json', '_errors.json')
        with open(error_file, 'w') as f:
            json.dump({"errors": errors}, f, indent=4)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8], help="List of temperatures for model conversation.")
    parser.add_argument("--repeat", type=int, default=1, help="The number of repetitions of the same query, exceed 1 only for image-only inference attack")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--input-json-path", type=str, required=True, help="Path to the instruct.json file")
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--output-json-path", type=str, required=True)
    args = parser.parse_args()
    main(args)
