from datasets import load_dataset
from tqdm import tqdm
import json
import os
import ast

OPTION_DICT = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",}

def mmmu(with_cot=False):
    mmmu_dataset = load_dataset('lmms-lab/MMMU')['validation']
    dataset = []
    for item in tqdm(mmmu_dataset, total=len(mmmu_dataset), desc="Loading MMMU..."):
        # print(item)
        prompt = ''
        if item["question_type"] == "multiple-choice":
            item["options"] = ast.literal_eval(item["options"])
            options = [f"Option {OPTION_DICT[i]}: {item['options'][i]}" for i in range(len(item["options"]))]
            prompt = '\n' + ", ".join(options) + '\n\n'
            if with_cot:
                prompt += " First provide an explanation of your decision-making process in at most one paragraph, and then provide your final answer out of the option."
        else:
            if with_cot:
                prompt = " First provide an explanation of your decision-making process in at most one paragraph, and then provide your final answer."
        if item["image_2"] is None:
            continue
        dataset.append(
            {
                "qid": item["id"],
                "question": item["question"] + prompt,
                "image": item["image_1"],
                "options": item["options"],
                "answer": item["answer"],
                "type": item["question_type"]
            }
        )

    return dataset

def naturalbench(with_cot=False):

    dataset = load_dataset("BaiqiL/NaturalBench")["train"]
    naturalbench = []

    SUFFIX_FOR_VQA = {
        "yes_no": " Please answer Yes or No.",
        "multiple_choice": " Please output the letter corresponding to the correct option.",
    }

    if with_cot:
        SUFFIX_FOR_VQA["yes_no"] += "First provide an explanation of your decision-making process in at most one paragraph, and then provide your final answer of YES or NO."
        SUFFIX_FOR_VQA["multiple_choice"] += "First provide an explanation of your decision-making process in at most one paragraph, and then provide your final answer of the letter corresponding to the correct option."
        
    for item in tqdm(dataset, desc="Loading NaturalBench..."):
        naturalbench.append(
            {
                "qid": str(item["Index"]) + '_q0_i0',
                "question": item["Question_0"] + SUFFIX_FOR_VQA[item["Question_Type"]],
                "image": item["Image_0"],
                "answer": item["Image_0_Question_0"],
                "type": item["Question_Type"]
            }
        )
        naturalbench.append(
            {
                "qid": str(item["Index"]) + '_q0_i1',
                "question": item["Question_0"] + SUFFIX_FOR_VQA[item["Question_Type"]],
                "image": item["Image_1"],
                "answer": item["Image_1_Question_0"],
                "type": item["Question_Type"],
            }
        )
        naturalbench.append(
            {
                "qid": str(item["Index"]) + '_q1_i0',
                "question": item["Question_1"] + SUFFIX_FOR_VQA[item["Question_Type"]],
                "image": item["Image_0"],
                "answer": item["Image_0_Question_1"],
                "type": item["Question_Type"],
            }
        )
        naturalbench.append(
            {
                "qid": str(item["Index"]) + '_q1_i1',
                "question": item["Question_1"] + SUFFIX_FOR_VQA[item["Question_Type"]],
                "image": item["Image_1"],
                "answer": item["Image_1_Question_1"],
                "type": item["Question_Type"],
            }
        )
    

    return naturalbench

def aokvqa(with_cot=False):
    if os.path.exists('datasets/aokvqa.json'):
        with open('datasets/aokvqa.json', 'r') as f:
            aokvqa = json.load(f)
        return aokvqa
    # Load the AOKVQA dataset
    img_dir = '/nas-ssd2/dataset/coco2017/val2017'

    with open('/nas-hdd/nsivaku/datasets/aokvqa/annotations/aokvqa_v1p0_val.json', 'r') as f:
        dataset = json.load(f)
        
    with open('/nas-hdd/nsivaku/datasets/aokvqa/annotations/val_id2file.json', 'r') as f:
        file_mapper = json.load(f)
    
    aokvqa = []
    for item in tqdm(dataset, desc="Loading AOKVQA..."):
        aokvqa.append(item)
        image_name = file_mapper[str(item["image_id"])]
        image_path = os.path.join(img_dir, image_name)
        if with_cot:
            aokvqa[-1]['question'] += " First provide an explanation of your decision-making process in at most one paragraph, and then provide your one-word final answer."
        aokvqa[-1]['image'] = image_path
    
    return aokvqa