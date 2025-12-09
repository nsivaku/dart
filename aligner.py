from models import Qwen
from omegaconf import DictConfig

def aligner(aligner_llm, expert_output, agent_output, disagreement):
    prompt = f"""You are an aligning agent. That means that you determine whether or not two different outputs are misaligned (0) or aligned (1). 
An expert output has provided a response to a disagreement between multiple different agents. Your task is to determine whether or not the expert output is aligned with a single agent's output.
Reason about the alignment of the agent output with the expert output. Then, output a 0 if the agent output is not aligned with the expert's output, and a 1 if it is aligned.

Output Format:
{{
    "reasoning": "Your reasoning for the alignment of the agent output with the expert output.",
    "alignment": "0 if the agent output is not aligned with the expert's output, and 1 if it is aligned."
}}

Disagreement: {disagreement}
Expert Output: {expert_output}
Agent Output: {agent_output}"""

    response = aligner_llm.run(prompt)
    return response