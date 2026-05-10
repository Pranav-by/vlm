import argparse
import os
import random
import json
import time
from PIL import Image
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import traceback

from transformers import StoppingCriteriaList
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="MiniGPT-4 Image Description")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--dataset-path", required=True, help="Path to the dataset directory containing images and filter_cap.json.")
    parser.add_argument("--output-json-path", required=True, help="Path to the output directory.")
    parser.add_argument("--num-beams", type=int, default=1, help="number of searching beams.")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8], help="List of temperatures for model conversation.")
    parser.add_argument("--repeat", type=int, default=1, help="The number of repetitions of the same query, exceed 1 only for image-only inference attack")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def initialize_model(cfg, gpu_id):
    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}
    print('Initializing Chat')
    model_config = cfg.model_cfg
    model_config.device_8bit = gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'cuda:{gpu_id}')
    CONV_VISION = conv_dict[model_config.model_type]
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(device=f'cuda:{gpu_id}') for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = Chat(model, vis_processor, device=f'cuda:{gpu_id}', stopping_criteria=stopping_criteria)
    print('Initialization Finished')
    return chat, CONV_VISION

def process_images(dataset_path, prompt, output_file, chat, CONV_VISION, num_beams, temperatures, repeat):
    directory = os.path.dirname(output_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    responses = []
    errors = []
    with open(os.path.join(dataset_path, 'filter_cap.json'), 'r') as file:
        data = json.load(file)
    count = 0
    for item in data['annotations']:
        start_time = time.time()
        image_id = item['image_id']
        caption = item['caption']
        img_path = os.path.join(dataset_path, 'image', f'{image_id}.jpg')
        try:
            img = Image.open(img_path)
            conversation_result = {
                "image_id": image_id,
            }
            for temperature in temperatures:
                conversation_result[f"conversations_{temperature}"] = []
                conversation_result[f"conversations_{temperature}"].append({"from": "human", "value": prompt})
                repeat_count = 0
                for _ in range(repeat): 
                    chat_state = CONV_VISION.copy()
                    img_list = []
                    llm_message = chat.upload_img(img, chat_state, img_list)
                    chat.encode_img(img_list)
                    chat.ask(prompt, chat_state)
                    llm_message = chat.answer(conv=chat_state,
                                            img_list=img_list,
                                            num_beams=num_beams,
                                            temperature=temperature,
                                            max_new_tokens=300,
                                            max_length=2000
                                        )[0]
                    repeat_count += 1
                    conversation_result[f"conversations_{temperature}"].append({"from": f"vlm_{repeat_count}", "value": llm_message})
                conversation_result[f"conversations_{temperature}"].append({"from": "ground truth", "value": caption})
            responses.append(conversation_result)
        except Exception as e:
            error_msg = f"Error processing image {image_id}: {str(e)}"
            print(error_msg)
            errors.append({'image_id': image_id, 'error': error_msg, 'traceback': traceback.format_exc()})
        finally:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            count += 1
            if count % 100 == 0:
                print(f"Finish conversation on {count} samples, process of last 100 samples takes {elapsed_time} seconds.")
                with open(output_file, 'w') as f:
                    json.dump(responses, f, indent=4)
                if errors:
                    error_file = output_file.replace('.json', '_errors.json')
                    with open(error_file, 'w') as f:
                        json.dump(errors, f, indent=4)
        
    with open(output_file, 'w') as f:
        json.dump({"responses": responses}, f, indent=4)
    print(f"Results are saved in {output_file}.")
    if errors:
        error_file = output_file.replace('.json', '_errors.json')
        with open(error_file, 'w') as f:
            json.dump({"errors": errors}, f, indent=4)

def main():
    args = parse_args()
    cfg = Config(args)
    chat, CONV_VISION = initialize_model(cfg, args.gpu_id)

    prompt = "Please provide a detailed description of the picture."
    directory = os.path.dirname(args.output_json_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    process_images(args.dataset_path, prompt, args.output_json_path, chat, CONV_VISION, args.num_beams, args.temperatures, args.repeat)

if __name__ == "__main__":
    main()


