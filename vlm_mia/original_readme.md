# Membership Inference Attacks Against Vision-Language Models

This repository contains the experimental code for the paper **"Membership Inference Attacks Against Vision-Language Models"**. Below are instructions for setting up the environment, training Vision-Language Models (VLMs), generating conversation outputs, and performing Membership Inference Attacks.

## VLM Training

This experiment focuses on two popular VLMs: [LLaVA](https://github.com/haotian-liu/LLaVA/tree/v1.0.1) and [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4). You will need to set up the environment, download pre-trained weights, split datasets, and train the models (instruction tuning stage only).

### LLaVA

To quickly set up your environment using Anaconda:

```bash
cd LLaVA
conda env create -f environment_llava.yml
conda activate llava
```

Then, follow the [LLaVA-Train](https://github.com/haotian-liu/LLaVA/tree/v1.0.1?tab=readme-ov-file#train) instructions to configure checkpoints and parameters.

Since this MIA targets the **Visual Instruction Tuning** stage, after preparing the Vicuna/LLaMA checkpoints, you can skip the [Pretrain](https://github.com/haotian-liu/LLaVA/tree/v1.0.1?tab=readme-ov-file#pretrain-feature-alignment) step and directly download the pre-trained projectors from [LLaVA MODEL_ZOO](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md#projector-weights) for Visual Instruction Tuning.

Download the instruction tuning data: [llava_instruct_158k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json) and COCO train2017 images [here](https://cocodataset.org/#download).

To distinguish between member and non-member data for MIA, split the `llava_instruct_158k.json` randomly. In our experiment, 80% of the data is used for training the model, and 20% is kept as non-member data. For Shadow Model Attacks, further divide the data into shadow-member, shadow-non-member, target-member, and target-non-member.

Next, follow the [Visual Instruction Tuning](https://github.com/haotian-liu/LLaVA/tree/v1.0.1?tab=readme-ov-file#visual-instruction-tuning) instructions to train the model using member data.

### MiniGPT-4

To quickly set up your environment using Anaconda:

```bash
cd MiniGPT-4
conda env create -f environment_minigptv.yml
conda activate minigptv
```

Then, follow the [MiniGPT4-Getting Started](https://github.com/haotian-liu/LLaVA/tree/v1.0.1?tab=readme-ov-file#visual-instruction-tuning) instructions to prepare the pretrained LLM weights and set the necessary variables in the model config files.

Refer to [Training of MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/MiniGPT4_Train.md) for training instructions. Since this MIA targets the **Visual Instruction Tuning** stage (Second fine-tuning stage), you can skip the pre-training stage and directly download the checkpoint [MiniGPT-4 (13B)](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link)  or [MiniGPT-4 (7B)](https://drive.google.com/file/d/1HihQtCEXUyBM1i9DQbaK934wW3TZi-h5/view?usp=share_link). Afterward, follow the Second fine-tuning stage instructions to prepare data and train the model, remembering to keep a portion of the data as non-member data.



## Conversation Generation

Next, use the trained models to query with member and non-member data. For Shadow Model Attacks, we have four groups of data to query: shadow-member, shadow-non-member, target-member, and target-non-member.

### LLaVA

Instead of using the scripts from [LLaVA-Demo](https://github.com/haotian-liu/LLaVA/tree/v1.0.1?tab=readme-ov-file#demo), run `conversation_llava.py` for batch conversation generation.

```bash
python conversation_llava.py --model-path "path to your trained model" --input-json-path "path to the input data in json format" --image-folder "path to corresponding image folder" --output-json "file name of the output data" --temperatures xx --repeat xx
```

Each input file corresponds to an output file in JSON format, and the input file of member/non-member data should match the structure of `llava_instruct_158k.json`.

### MiniGPT-4

Before querying the model, refer to [MiniGPT4-Installation](https://github.com/Vision-CAIR/MiniGPT-4?tab=readme-ov-file#installation) to set the parameters in the config files, ensuring to point to the pretrained checkpoint you just trained. Then, use `conversation_minigpt4.py` to generate batch queries.

```bash
python conversation_llava.py --cfg-path "path to the evaluation configuration file" --dataset-path "Path to the dataset directory containing images and filter_cap.json" --image-folder "path to corresponding image folder" --output-json-path "path to the output data" --temperatures xx --repeat xx
```

For all attacks except Image-only Attack, set `repeat` to 1. For Image-only Attack, set it to a value greater than 1 (recommended between 5-10).

For Reference Attack and Image-only Attack, set temperatures to a single value (e.g., 0.1). If you're also conducting Target-only Attack, include two values with a significant difference (e.g., [0.1, 1.5]). For Shadow Model Attack, use more temperature values.

## Similarity Calculation

Once you have obtained conversation outputs, calculate the text similarity between member and non-member data for the subsequent attack.

First, create a new environment:

```bash
conda env create -f vlm_mia.yml
conda activate vlm_mia
```

For the first three attacks, use `similarity_with_ground_truth.py` to compute the similarity:

```bash
python similarity_with_ground_truth.py --conversation_json_path "path to the conversation output file" --similarity_json_path "path to the similarity data" --temperatures xx
```

For Image-only Attack, use `similarity_with_repeating_generation.py`:

```bash
python similarity_with_repeating_generation.py --conversation_json_path "path to the conversation output file" --similarity_json_path "path to the similarity data" --temperatures xx --repeating_num xx
```

The temperatures and repeating_num should match the parameters used in the conversation generation.

The similarity calculation code includes an OpenAI API method. If you don’t need it, you can comment out the relevant part.

## Inference Attack

Each attack type has its corresponding code file. Run the appropriate script to obtain the attack success rate.

```bash
python shadow_model_inference.py --shadow_member_similarity_file "path to shadow_member_similarity_file" --shadow_non_member_similarity_file "path to shadow_non_member_similarity_file" --target_member_similarity_file "path to target_member_similarity_file" --target_non_member_similarity_file "path to target_non_member_similarity_file" --granularity xx --temperatures xx --similarity_metric xx --with_variance

python reference_non_member_inference.py --member_similarity_file "path to member_similarity_file" --non_member_similarity_file "path to non_member_similarity_file" --granularity xx --temperature xx --similarity_metric xx

python reference_non_member_inference.py --member_similarity_file "path to member_similarity_file" --non_member_similarity_file "path to non_member_similarity_file" --granularity xx --temperature xx --similarity_metric xx

python target_only_inference.py --member_similarity_file "path to member_similarity_file" --non_member_similarity_file "path to non_member_similarity_file" --granularity xx --temperature_high xx --temperature_low xx --similarity_metric xx

python image_only_inference.py --member_similarity_file "path to member_similarity_file" --non_member_similarity_file "path to non_member_similarity_file" --granularity xx --temperature xx --similarity_metric xx
```

