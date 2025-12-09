# DART: Leveraging Multi-Agent Disagreement for Tool Recruitment in Multimodal Reasoning

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2512.07132)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Nithin Sivakumaran](https://nsivaku.github.io/) | [Justin Chih-Yao Chen](https://dinobby.github.io/) | [David Wan](https://meetdavidwan.github.io/) | [Yue Zhang](https://zhangyuejoslin.github.io/) | [Jaehong Yoon](https://jaehong31.github.io/) | [Elias Stengel-Eskin](https://esteng.github.io/) | [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

![Your Image](images/main_figure.png)
Figure: Previous work has explored using (A) multiple agents in  debate to refine their reasoning, but this approach is limited to the abilities of the agents. Alternatively, some methods employ a (B) top‑down LLM agent that invokes vision tools, yet they plan tool usage based solely on the question and overlook the visual information itself.
In our method (C), we facilitate a discussion among multiple agents with targeted intervention from a pool of vision tools. These tools address disagreements detected in a debate of VLM agents, with their specialized vision outputs and agreement scores being used for future discussion.

## Installation
We create an environment with Python 3.9 and install the required packages.

```bash
conda create --name dart python=3.9
conda activate dart
pip install -r requirements.txt
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.45 --force-reinstall --no-cache-dir
pip install flash-attn --no-build-isolation
```

## Dataset Setup

**NaturalBench** and **MMMU** are automatically downloaded through the HuggingFace datasets package in ``dataset.py``.

To set up **A-OKVQA**:
1. Download the compressed annotation file: [https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz](https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz)
2. Download MSCOCO 2017 validation set images: [https://cocodataset.org/#download](https://cocodataset.org/#download)
3. We provide an image ID to file name mapping to download : https://drive.google.com/file/d/1f2mXf06iMoUVDHIr3BWtxFY_L113WLHj/view?usp=sharing
3. Update the corresponding file paths in the `aokvqa()` function in `dataset.py`.

## Run evaluation

To start an evaluation, run `dart.py`.

```bash
python dart.py \
  --cfg configs/default.yaml \
  --exp_name {exp_name} \
  --output_file {output_file}.json
```

### Arguments

* `--cfg` (str, default: `configs/default.yaml`): Path to configuration file in `configs/` directory.
* `--output_file` (str, default: `default.json`): Name of the final JSON results file.
* `--exp_name` (str, default: `aokvqa`): Experiment name. Used only to organize results:
  * Outputs are saved to `results/<exp_name>/<timestamp>/`.
* `--resume` (default: 0): Index to resume from (0-based)

## Citation

```bibtex
@article{sivaku2025dart,
      title={DART: Leveraging Multi-Agent Disagreement forTool Recruitment in Multimodal Reasoning}, 
      author={Nithin Sivakumaran and Justin Chih-Yao Chen and David Wan and Yue Zhang and Jaehong Yoon and Elias Stengel-Eskin and Mohit Bansal},
      journal={arXiv preprint arXiv:2512.07132},
      year={2025}
}
```



