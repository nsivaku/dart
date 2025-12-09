import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
from datetime import datetime
import os
import json
import traceback
from collections import OrderedDict
import logging
from PIL import Image
import re
import json_repair

from models import *
from modules import expert_functions
from dataset import mmmu, naturalbench, aokvqa
from aligner import aligner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_DICT = {"minicpm": MiniCPM_o, "minicpm_o": MiniCPM_o, "qwenvl": QwenVL, "ovis": Ovis2}
DATASET_DICT = {"mmmu": mmmu, "naturalbench": naturalbench, "aokvqa": aokvqa}

answer_form = {"mmmu": "one of the answer choices", "naturalbench": "YES or NO", "aokvqa": "one word or phrase"}



field_pattern = re.compile(
    r"(Answer|Reasoning|Confidence):\s*(.*?)"
    r"(?=\n(?:Answer|Reasoning|Confidence):|\Z)",
    re.DOTALL | re.IGNORECASE,
)

def parse_qa_response(text: str):
    """
    Parse text containing any order of:
        Answer: ...
        Reasoning: ...
        Confidence: ...

    Returns (answer, reasoning, confidence_float).
    """
    data = {}
    for name, value in field_pattern.findall(text):
        key = name.lower()  # "answer", "reasoning", "confidence"
        data[key] = value.strip()

    if not {"answer", "reasoning", "confidence"} <= data.keys():
        raise ValueError("Response not in expected format")

    answer = data["answer"]
    reasoning = data["reasoning"]
    confidence = float(data["confidence"])

    return answer, reasoning, confidence

def get_expert_requirements(planner_data):
    if not planner_data:
        return []
    
    experts = planner_data.get('experts', [])
    inputs = planner_data.get('inputs', {})
    
    expert_requirements = []
    for expert in experts:
        expert_input = inputs.get(expert, {})
        expert_requirements.append({
            'expert_name': expert,
            'disagreement': expert_input.get('disagreement', ''),
            'justification': expert_input.get('justification', ''),
            'arguments': expert_input.get('arguments', [])
        })


def execute_expert(expert_name, image_path, question, expert_requirements):
    if expert_name not in expert_functions:
        return f"Expert '{expert_name}' not available. Available experts: {list(expert_functions.keys())}"
    
    try:
        expert_func = expert_functions[expert_name]
        
        # Prepare inputs based on expert type
        if expert_name == 'ocr':
            result = expert_func(image_path, plain_text=True)
        elif expert_name in ['grounder']:
            arguments = expert_requirements.get('arguments', [])
            text_prompt = ' '.join(arguments) if arguments else question
            result = expert_func(image_path, text_prompt)
        elif expert_name in ['captioner', 'reasoner', 'attribute']:
            prompt = f"Question: {question}\nDisagreement: {expert_requirements.get('disagreement', '')}"
            result = expert_func(image_path, prompt)
        else:
            result = expert_func(image_path)
        
        return result
        
    except Exception as e:
        return f"Error executing expert '{expert_name}': {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--cfg", default="configs/default.yaml", type=str)
    parser.add_argument("--output_file", default="default.json", type=str)
    parser.add_argument("--exp_name", default="aokvqa", type=str)
    parser.add_argument("--resume", default=0, type=int)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    output_file = args.output_file
    exp_name = args.exp_name
    dataset_name = cfg.dataset.name
    
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y%m%d_%H%M%S")
    output_folder = f'results/{exp_name}/{folder_name}' if exp_name else f'outputs/{folder_name}'
    file_output = os.path.join(output_folder, output_file)
    answers_file = os.path.expanduser(file_output)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    cfg = OmegaConf.load(args.cfg)
    agents = [MODEL_DICT[agent.split('_')[0]](cfg=cfg.agents[agent]) for agent in cfg.agents.keys()]
    logger.info(f"Agents Loaded: {', '.join([a.__class__.__name__ for a in agents])}")

    annotations = DATASET_DICT[dataset_name]()
    planner = Qwen(cfg=cfg.llm)
    logger.info(f"Dataset {dataset_name} loaded.")
    
    index = 0
    json_file = []
    with tqdm(total=len(annotations)) as pbar:
        for data in annotations:
            if index < args.resume:
                index += 1
                pbar.update(cfg.debate.batch_size)
                continue
            try:
                image = data['image']
                if type(image) == str:
                    image = Image.open(image)

                question_id = data['qid']
                question = data['question']

                if dataset_name == "mmmu":
                    question = f'{question}\n{data["options"]}'
                    
                json_data = data

                json_data['prompt'] = template = """{}

Answer the question with only {}, then provide step-by-step reasoning for the answer. Finally, provide a confidence score from 0-1 for your answer (0 meaning not confident at all, 1 meaning complete confidence)

You must output in the following format:
Answer: [answer]
Reasoning: [reasoning]
Confidence: [confidence]
""".format(question, answer_form[dataset_name])
                vqa_prompt = json_data['prompt']
                
                qa_outputs = [agent.run(image, vqa_prompt).strip() for agent in agents]

                debate_prompt = """Carefully review the following solutions from other agents as additional information, and provide your own answer and step-by-step reasoning to the question.
                
Clearly state which point of view you agree or disagree with and why. Each reasoning and answer is equally valid. Do not put extra weight or emphasis on longer reasonings. Analyze each justification objectively and do not overthink the answers.

{}

Output your response in the following format:

Reasoning: [reasoning for your answer to the original question based on the other agents responses]
Answer: [your answer to the orignal question]

You must follow the provided format no matter what. This rule is unbreakable."""

                current_round = {}
                json_data['r0'] = {}

                for idx, response in enumerate(qa_outputs):
                    try:
                        answer, reasoning, confidence = parse_qa_response(response)
                        answer_norm = answer.lower()

                        current_round.setdefault(answer_norm, []).append(reasoning)
                        json_data['r0'][f'a{idx+1}'] = {
                            'answer': answer_norm,
                            'reasoning': reasoning,
                            'confidence': confidence,
                        }
                    except Exception:
                        logger.error(f'Invalid response for Round 0 Agent {idx+1}')
                        json_data['r0'][f'a{idx+1}'] = {'response': response}

                current_round = OrderedDict(
                    sorted(current_round.items(), key=lambda x: len(x[1]))
                )

                planning_prompt = """Here was the initial prompt: {}\n\nCarefully review the following solutions from other agents for the provided question. Now, analyze what disagreements are occuring between the different agents.

{}

You have the ability to call on different experts, each with their own specialized capabilities. Based on the disagreements you observed, pick out the set of experts (could be just one) that would be best equipped to solve all the disagreements.

Here are all the experts, their inputs, and their capabilities/usage:

"spatial" (input: list. objects that have confused spatial relations) - Has perfect understanding of spatial relations between objects. Use this when agents are unsure about the placement of items in a scene.
"ocr" (input: none)- Can correctly read all text in an image. Use this when agents have differing views on what the text is in an image.
"grounder" (input: list. objects you are trying to find) - Will find any object if it is an image, otherwise it will return nothing. Use this when agents are not agreeing on what's present in an image.
"detector" (input: none) - Will provide a list of objects in the image, their counts, and their bounding boxes. Only use this when agents are differing in their counts of objects in an image.
"captioning" (input: list) - Can give a detailed description of what's going in the image relevant to the question. Use this when agents might need a better idea of the general scene or descriptions of specific objects.
"reasoning" (input: none) - Has better world knowledge and advanced reasoning capabilities about what might be going on in an image. Use this when agents are confused or conflicting in their inferences about the scene.
"attribute" (input: list. objects you want attributes for) - Will give information on different features of objects in the image, including color, properties, catgories, and more. Use this when agents are confused about the features of relevant objects and need many surface level features.

Output the expert(s) you need to resolve the disagreements to answer the original question: {}. This should be a JSON with this format like this. You can call more or less experts than this as needed:

{{
    "experts": ["grounder", "attribute", "ocr"],
    "inputs": {{
        "grounder": {{
            "disagreement": "Agent 1 mentioned that there is a cat, but Agent 2 said there is no cat and instead said it is a dog.",
            "justification": "The grounder will help resolve the disagreement about the presence of a cat or dog in the image.",
            "arguments": ["cat", "dog"]
        }},
        "attribute": {{
            "disagreement": "Agent 1 said the flower is red, Agent 2 said it is orange, and Agent 3 did not specifically mention anything about the flower. There also was confusion about the details of the car.",
            "justification": "The attribute expert will help resolve the disagreement about the color of the flower and provide details about the car.",
            "arguments": ["flower", "car"]
        }},
        "ocr": {{
            "disagreement": "Agents conflict on the text they see in the image.",
            "justification": "The OCR expert will see the text in the image and resolve the disagreement.",
            "arguments": []
        }}
    }}
}}

Now give the expert output in the given format for the previous agent solutions and the question provided above. Do not be redudant on disagreements unless the expert is adding new information that better resolves the disagreement. If all agents agree on their overall answer, you do not need to call any experts. If there are no disagreement on the overall answer, output an empty JSON object like this: {{}}."""

                agent_solutions = ""
                for answer, reasonings in current_round.items():
                    agent_solutions += f'There are {len(reasonings)} agents that think the answer is {answer}.\n'.replace('are 1 agents that think', 'is 1 agent that thinks')
                    for agent_reasoning in reasonings:
                        agent_solutions += f'One agent solution: {answer} - {agent_reasoning}\n'
                    agent_solutions += '\n'
                
                agent_solutions = agent_solutions.strip()
                planning_prompt = planning_prompt.format(vqa_prompt, agent_solutions, question)
                planner_output = planner.run(planning_prompt).strip()
                planner_output = json_repair.loads(planner_output)
                json_data['experts_recruited'] = planner_output['experts']
                json_data['expert_justifications'] = planner_output['inputs']
                expert_requirements = get_expert_requirements(planner_output)

                expert_results = {}

                for req in expert_requirements:
                    expert_name = req['expert_name']
                    result = execute_expert(expert_name, image, question, req)
                    expert_results[expert_name] = result

                json_data['expert_results'] = expert_results

                expert_outputs = ""
                for expert_name, result in expert_results.items():
                    expert_outputs += f"{expert_name}: {result}\n"

                alignment_scores = {0: [0 for _ in range(len(expert_results.keys()))], 1: [0 for _ in range(len(expert_results.keys()))], 2: [0 for _ in range(len(expert_results.keys()))]}
                for agent in json_data['r0'].keys():
                    for idx, expert_result in enumerate(expert_results.values()):
                        alignment_scores[0][idx] += json_repair.loads(aligner(planner, expert_result, json_data['r0'][agent]['reasoning'], json_data['r0'][f'a{idx+1}']['disagreement']))['alignment']
                        alignment_scores[1][idx] += json_repair.loads(aligner(planner, expert_result, json_data['r0'][agent]['reasoning'], json_data['r0'][f'a{idx+1}']['disagreement']))['alignment']
                        alignment_scores[2][idx] += json_repair.loads(aligner(planner, expert_result, json_data['r0'][agent]['reasoning'], json_data['r0'][f'a{idx+1}']['disagreement']))['alignment']
                alignment_scores = {0: sum(alignment_scores[0]) / len(alignment_scores[0]), 1: sum(alignment_scores[1]) / len(alignment_scores[1]), 2: sum(alignment_scores[2]) / len(alignment_scores[2])}
                alignment_scores = {k: round(v, 2) for k, v in alignment_scores.items()}
                json_data['r0']['a1']['expert_alignment'] = alignment_scores[0]
                json_data['r0']['a2']['expert_alignment'] = alignment_scores[1]
                json_data['r0']['a3']['expert_alignment'] = alignment_scores[2]

                current_round = {}
                json_data['r0'] = {}

                for idx, response in enumerate(qa_outputs):
                    try:
                        answer, reasoning, confidence = parse_qa_response(response)
                        answer_norm = answer.lower()

                        json_data['r0'][f'a{idx+1}'] = {
                            'answer': answer_norm,
                            'reasoning': reasoning,
                            'confidence': confidence,
                            'expert_alignment': alignment_scores[idx],
                        }

                        current_round.setdefault(answer_norm, []).append({
                            'reasoning': reasoning,
                            'confidence': confidence,
                            'expert_alignment': alignment_scores[idx],
                        })

                    except Exception:
                        logger.error(f'Invalid response for Round 0 Agent {idx+1}')
                        json_data['r0'][f'a{idx+1}'] = {'response': response}

                current_round = OrderedDict(
                    sorted(current_round.items(), key=lambda x: len(x[1]))
                )

                agent_solutions = ""
                for answer, agents in current_round.items():
                    header = f"There are {len(agents)} agents that think the answer is {answer}.\n"
                    header = header.replace("are 1 agents that think", "is 1 agent that thinks")
                    agent_solutions += header

                    for agent in agents:
                        reasoning = agent["reasoning"]
                        confidence = agent.get("confidence")
                        expert_alignment = agent.get("expert_alignment")

                        agent_solutions += (
                            f"One agent solution: {answer} - {reasoning} "
                            f"(confidence: {confidence:.3f}, expert_agreement: {expert_alignment:.3f})\n"
                        )

                    agent_solutions += "\n"
                agent_solutions = agent_solutions.strip()

                debate_prompt = """Carefully review the following solutions from other agents as additional information, and provide your own answer and step-by-step reasoning to the question.
                    
Clearly state which point of view you agree or disagree with and why. Each reasoning and answer is equally valid. Do not put extra weight or emphasis on longer reasonings. Analyze each justification objectively and do not overthink the answers.

{}

We also have these experts that were used to resolve disagreements among the agents: {}

Here are their outputs:
{}

Since the experts have greater knowledge in their domains, you should defer to their outputs when reasoning about the question. If something in the agent solution contradicts the experts, assume that the expert is right. Carefully analyze the expert outputs and the agent solutions, then provide your own answer and step-by-step reasoning to the question.

Output your response in the following format:

Reasoning: [reasoning for your answer to the original question based on the other agents responses]
Answer: [your new answer to the orignal question]
Confidence: [confidence score for your new answer]

Some example outputs:

Reasoning: I agree with Agent 1's reasoning that the answer is A because it aligns with the expert's analysis. The expert confirmed that the condition described in the question is consistent with the findings in the image.
Answer: Option A
Confidence: 0.95

Reasoning: I disagree with Agent 2's reasoning that the answer is B. The expert's analysis suggests that the it is more likely to be A, as it matches the what is described in the question and the findings in the image.
Answer: Option A
Confidence: 0.90

Reasoning: I agree with Agent 1 and 3's reasoning that the answer is C. Although the expert's analysis supports this conclusion, the evidence provided by the expert tool is not compelling enough to go against the other agents' reasoning.
Answer: Option C
Confidence: 0.80

You must follow the provided format no matter what. This rule is unbreakable.""".format(agent_solutions, ', '.join(expert_results.keys()), expert_outputs)

                qa_outputs = [agent.run(image, debate_prompt).strip() for agent in agents]

                alignment_scores = {0: [0 for _ in range(len(expert_results.keys()))], 1: [0 for _ in range(len(expert_results.keys()))], 2: [0 for _ in range(len(expert_results.keys()))]}
                for agent in json_data['r1'].keys():
                    for idx, expert_result in enumerate(expert_results.values()):
                        alignment_scores[0][idx] += json_repair.loads(aligner(planner, expert_result, json_data['r1'][agent]['reasoning'], json_data['r1'][f'a{idx+1}']['disagreement']))['alignment']
                        alignment_scores[1][idx] += json_repair.loads(aligner(planner, expert_result, json_data['r1'][agent]['reasoning'], json_data['r1'][f'a{idx+1}']['disagreement']))['alignment']
                        alignment_scores[2][idx] += json_repair.loads(aligner(planner, expert_result, json_data['r1'][agent]['reasoning'], json_data['r1'][f'a{idx+1}']['disagreement']))['alignment']
                alignment_scores = {0: sum(alignment_scores[0]) / len(alignment_scores[0]), 1: sum(alignment_scores[1]) / len(alignment_scores[1]), 2: sum(alignment_scores[2]) / len(alignment_scores[2])}
                alignment_scores = {k: round(v, 2) for k, v in alignment_scores.items()}
                json_data['r1']['a1']['expert_alignment'] = alignment_scores[0]
                json_data['r1']['a2']['expert_alignment'] = alignment_scores[1]
                json_data['r1']['a3']['expert_alignment'] = alignment_scores[2]

                current_round = {}
                json_data['r1'] = {}

                for idx, response in enumerate(qa_outputs):
                    try:
                        answer, reasoning, confidence = parse_qa_response(response)
                        answer_norm = answer.lower()

                        json_data['r1'][f'a{idx+1}'] = {
                            'answer': answer_norm,
                            'reasoning': reasoning,
                            'confidence': confidence,
                            'expert_alignment': alignment_scores[idx],
                        }

                        current_round.setdefault(answer_norm, []).append({
                            'reasoning': reasoning,
                            'confidence': confidence,
                            'expert_alignment': alignment_scores[idx],
                        })

                    except Exception:
                        logger.error(f'Invalid response for Round 1 Agent {idx+1}')
                        json_data['r1'][f'a{idx+1}'] = {'response': response}

                current_round = OrderedDict(
                    sorted(current_round.items(), key=lambda x: len(x[1]))
                )

                agent_solutions = ""
                for answer, agents in current_round.items():
                    header = f"There are {len(agents)} agents that think the answer is {answer}.\n"
                    header = header.replace("are 1 agents that think", "is 1 agent that thinks")
                    agent_solutions += header

                    for agent in agents:
                        reasoning = agent["reasoning"]
                        confidence = agent.get("confidence")
                        expert_alignment = agent.get("expert_alignment")

                        agent_solutions += (
                            f"One agent solution: {answer} - {reasoning} "
                            f"(confidence: {confidence:.3f}, expert_agreement: {expert_alignment:.3f})\n"
                        )

                    agent_solutions += "\n"
                agent_solutions = agent_solutions.strip()

                aggregator_prompt = """You are an aggregator model that will be given an image, question, and set of answer choices. Your tasks is to select the best final answer for the question. You will also be given various sources of information to help inform your decision. Each answer was generated by a different agent, and these agent will provide their reasoning for why they gave their answer. You will also be given tool outputs from expert models that directly relate to the question and were used to resolve disagreement among the answering agents. Feel free to defer to these expert tool outputs if the answers contradict the info from the tools.

The question is: {}

Here are the different agent answers:
{}

Here are the tool outputs:
{}

Now based on the provided information, provide your step-by-step reasoning for selecting the best, most correct answer to the question provided. Then, give your confidence in your selected answer.

Output in the following format:

Reasoning: [reasoning]
Answer: [answer]
Confidence: [confidence]
"""

                agg_prompt = aggregator_prompt.format(
                    question,
                    agent_solutions,
                    expert_outputs
                )

                # aggregator is the same model as the first agent (Ovis)
                agg_output = agents[0].run(image, agg_prompt).strip()

                answer, reasoning, confidence = parse_qa_response(agg_output)
                answer_norm = answer.lower()
                json_data['r1']['aggregator'] = {
                    'answer': answer_norm,
                    'reasoning': reasoning,
                    'confidence': confidence,
                }

                json_file.append(json_data)
                if index % 5 == 0:
                    with open(f'{args.resume}_expert.json', 'w') as f:
                        json.dump(json_file, f, indent=4)
                index += 1
            except:
                traceback.print_exc()
            pbar.update(cfg.debate.batch_size)

        
        with open(f'{args.resume}_expert.json', 'w') as f:
            json.dump(json_file, f, indent=4)

if __name__ == "__main__":
    main()
