# OpenChat: Advancing Open-source Language Models with Mixed-Quality Data

<div align="center">
  <img src="assets/logo_new.png" style="width: 65%">
</div>

<p align="center">
  <a href="https://openchat.team">💻Online Demo</a> |
  <a href="https://huggingface.co/openchat">🤗Huggingface</a> |
  <a href="https://arxiv.org/pdf/2309.11235.pdf">📃Paper</a> |
  <a href="https://discord.gg/pQjnXvNKHY">💭Discord</a> 
</p>

- OpenChat is an innovative library of **open-source language models**, fine-tuned with [**C-RLFT**](https://arxiv.org/pdf/2309.11235.pdf) - a strategy inspired by offline reinforcement learning.
- Our models learn from mixed-quality data without preference labels, delivering exceptional performance on par with `ChatGPT`, even with a `7B` model which can be run on a **consumer GPU (e.g. RTX 3090)**.
- Despite our simple approach, we are committed to developing a high-performance, commercially viable, open-source large language model, and we continue to make significant strides toward this vision.

[![DOI](https://zenodo.org/badge/645397533.svg)](https://zenodo.org/badge/latestdoi/645397533)

# ✨ News

 - [2024/05/22] We released the Llama-3 based version [OpenChat 3.6 20240522](https://huggingface.co/openchat/openchat-3.6-8b-20240522), outperforming official Llama 3 8B Instruct and open-source finetunes/merges.

- [2024/01/06] We released the second update, [OpenChat 3.5 0106](openchat/openchat-3.5-0106), further improved coding and overall performance 🏆.

- [2023/12/10] We released the first update, [OpenChat 3.5 1210](openchat/openchat-3.5-1210), improved coding by 15 points 🚀.

- [2023/11/01] We released the [OpenChat-3.5-7B](https://huggingface.co/openchat/openchat_3.5) model, surpassing ChatGPT on various benchmarks 🔥.

- [2023/09/21] We released our paper [OpenChat: Advancing Open-source Language Models with Mixed-Quality Data](https://arxiv.org/pdf/2309.11235.pdf).
  
<details>
  <summary>Read more</summary>
  
- [2023/09/03] We released the [OpenChat V3.2 SUPER]([#models](https://huggingface.co/openchat/openchat_v3.2_super)) model.

- [2023/08/04] We have launched an [Online Demo](https://openchat.team) featuring the latest version, OpenChat 3.2.

- [2023/07/30] We are thrilled to introduce the [OpenChat V3 model series](#models), based on Llama 2, and now available for free for commercial use!

- [2023/07/07] We released the [OpenChat V2 model series](#legacy-models).

- [2023/07/01] We released the [OpenChat V1 model series](#legacy-models).
</details>

# 🏷️ Benchmarks - OpenChat 3.6

<div align="center">
  <img src="https://raw.githubusercontent.com/imoneoi/openchat/master/assets/benchmarks-openchat-3.6-20240522.svg" style="width: 95%;">
</div>


<details>
  <summary>Reproducing benchmarks</summary>

Note: Please run the following commands at the base directory of this repository.

```bash
python -m ochat.evaluation.run_eval --condition "GPT4 Correct" --model openchat/openchat-3.6-8b-20240522 --eval_sets fs_cothub/mmlu fs_cothub/gsm8k fs_cothub/math
python -m ochat.evaluation.run_eval --condition "GPT4" --model openchat/openchat-3.6-8b-20240522 --eval_sets zs/gpqa
```

HumanEval is run using the official [EvalPlus repository](https://github.com/evalplus/evalplus).
</details>

# 🏷️ Benchmarks - OpenChat 3.5

| Model                 | # Params | Average  | MT-Bench     | HumanEval       | BBH MC   | AGIEval  | TruthfulQA    | MMLU         | GSM8K        | BBH CoT     |
|-----------------------|----------|----------|--------------|-----------------|----------|----------|---------------|--------------|--------------|-------------|
| **OpenChat-3.5-0106** | **7B**   | **64.5** | 7.8          | **71.3**        | **51.5** | **49.1** | 61.0          | 65.8         | **77.4**     | 62.2        |
| ChatGPT (March)*      | ???B     | 61.5     | **7.94**     | 48.1            | 47.6     | 47.1     | 57.7          | **67.3**     | 74.9         | **70.1**    |
|                       |          |          |              |                 |          |          |               |              |              |             |
| OpenHermes 2.5        | 7B       | 59.3     | 7.54         | 48.2            | 49.4     | 46.5     | 57.5          | 63.8         | 73.5         | 59.9        |
| OpenOrca Mistral      | 7B       | 52.7     | 6.86         | 38.4            | 49.4     | 42.9     | 45.9          | 59.3         | 59.1         | 58.1        |
| Zephyr-β^             | 7B       | 34.6     | 7.34         | 22.0            | 40.6     | 39.0     | 40.8          | 39.8         | 5.1          | 16.0        |
| Mistral               | 7B       | -        | 6.84         | 30.5            | 39.0     | 38.0     | -             | 60.1         | 52.2         | -           |
| Open-source SOTA**    | 13B-70B  | 61.4     | 7.71         | 73.2            | 49.7     | 41.7     | 62.3          | 63.7         | 82.3         | 41.4        |
|                       |          |          | WizardLM 70B | WizardCoder 34B | Orca 13B | Orca 13B | Platypus2 70B | WizardLM 70B | MetaMath 70B | Flan-T5 11B |

<details>
  <summary>Evaluation details</summary>
*: ChatGPT (March) results are from GPT-4 Technical Report, Chain-of-Thought Hub, and our evaluation.

^: Zephyr-β often fails to follow few-shot CoT instructions, likely because it was aligned with only chat data but not trained on few-shot data.

 **: Mistral and Open-source SOTA results are taken from reported results in instruction-tuned model papers and official repositories.

All models are evaluated in chat mode (e.g. with the respective conversation template applied). All zero-shot benchmarks follow the same setting as in the AGIEval paper and Orca paper. CoT tasks use the same configuration as Chain-of-Thought Hub, HumanEval is evaluated with EvalPlus, and MT-bench is run using FastChat. To reproduce our results, follow the instructions below.
</details>

<details>
  <summary>Reproducing benchmarks</summary>

Reasoning and Coding:

Note: Please run the following commands at the base directory of this repository.

```bash
python -m ochat.evaluation.run_eval --condition "GPT4 Correct" --model openchat/openchat-3.5-0106 --eval_sets coding fs_cothub/bbh fs_cothub/mmlu zs/agieval zs/bbh_mc_orca zs/truthfulqa_orca
python ochat/evaluation/view_results.py
python ochat/evaluation/convert_to_evalplus.py
```

Then all humaneval code samples are placed in `ochat/evaluation/evalplus_codegen`. Use the following command to evaluate an individual code sample named `samples.jsonl` using Docker as a sandbox.

```bash
docker run -v $(pwd):/app ganler/evalplus:latest --dataset humaneval --samples samples.jsonl
```

Mathematical Reasoning:

Note: Please run the following commands at the base directory of this repository.

```bash
python -m ochat.evaluation.run_eval --condition "Math Correct" --model openchat/openchat-3.5-0106 --eval_sets fs_cothub/gsm8k zs/math
python ochat/evaluation/view_results.py
```

MT-Bench:

Please first launch a local API server, then download FastChat and run the following commands.

Note: Due to non-zero temperature and GPT-4 API changes over time, there might be variations in the results.

```bash
cd fastchat/llm_judge
python gen_api_answer.py --model openchat-3.5-0106 --max-tokens 4096 --parallel 128 --openai-api-base http://localhost:18888/v1
python gen_judgment.py --model-list openchat-3.5-0106 --parallel 8 --mode single
```

</details>

## 🎇 Comparison with [X.AI Grok](https://x.ai/)

🔥 OpenChat-3.5-0106 (7B) now outperforms Grok-0 (33B) on **all 4 benchmarks** and Grok-1 (???B) on average and **3/4 benchmarks**.

|                       | License     | # Param | Average  | MMLU   | HumanEval | MATH     | GSM8k    |
|-----------------------|-------------|---------|----------|--------|-----------|----------|----------|
| **OpenChat-3.5-0106** | Apache-2.0  | **7B**  | **61.0** | 65.8   | **71.3**  | **29.3** | **77.4** |
| Grok-0                | Proprietary | 33B     | 44.5     | 65.7   | 39.7      | 15.7     | 56.8     |
| Grok-1                | Proprietary | ???B    | 55.8     | **73** | 63.2      | 23.9     | 62.9     |

# ⬇️ Installation
> [!NOTE]
> Need [`pytorch`](https://pytorch.org/get-started/locally/#start-locally) to run OpenChat

## pip

```bash
pip3 install ochat
```
> [!IMPORTANT]
> If you are facing package compatibility issues with pip, try the conda method below or check [this issue](https://github.com/imoneoi/openchat/issues/41)

## conda

```bash
conda create -y --name openchat python=3.11
conda activate openchat

pip3 install ochat
```

## Windows (WSL 1.x, Ubuntu-22.04)

```bash
sudo apt update
sudo apt install build-essential

sudo apt install -y curl
curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh

# Restart WSL terminal if the following conda command does not work

conda create -y --name openchat python=3.11
conda activate openchat

pip3 install ochat
```

## From source

<details>
  <summary>Clone this repo and install openchat from source in editable mode</summary>

```bash
git clone https://github.com/imoneoi/openchat
cd openchat

pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e .  # Editable mode, you can make changes in this cloned repo
```
</details>

# 🚀 Deploying API server

⚡ Our API server is ready for production use and compatible with the OpenAI API protocol. It is highly optimized with vLLM and can dynamically batch requests.

📎 Note: For 20 series or older GPUs that do not support `bfloat16`, add `--dtype float16` to the server args.

### For a single GPU (e.g. RTX 3090, 4090)

```bash
python -m ochat.serving.openai_api_server --model openchat/openchat-3.5-0106
```

### For multiple GPUs (tensor parallel)

```bash
# N is the number of tensor parallel GPUs
python -m ochat.serving.openai_api_server --model openchat/openchat-3.5-0106 --engine-use-ray --worker-use-ray --tensor-parallel-size N
```

use `-h` to see more settings
```bash
python -m ochat.serving.openai_api_server --model openchat/openchat-3.5-0106 -h
```

<details>
  <summary>Deploy as online service</summary>

If you want to deploy the server as an online service, you can use `--api-keys sk-KEY1 sk-KEY2 ...` to specify allowed API keys and `--disable-log-requests --disable-log-stats --log-file openchat.log` for logging only to a file. For security purposes, we recommend using an [HTTPS gateway](https://fastapi.tiangolo.com/es/deployment/concepts/#security-https) in front of the server.

</details>

## Request example

Once started, the server listens at `localhost:18888` for requests and is compatible with the [OpenAI ChatCompletion API specifications](https://platform.openai.com/docs/api-reference/chat). 

💡 **Default Mode (GPT4 Correct)**: Best for coding, chat and general tasks

```bash
curl http://localhost:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openchat_3.5",
    "messages": [{"role": "user", "content": "You are a large language model named OpenChat. Write a poem to describe yourself"}]
  }'
```

🧮 **Mathematical Reasoning Mode**: Tailored for solving math problems

```bash
curl http://localhost:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openchat_3.5",
    "condition": "Math Correct",
    "messages": [{"role": "user", "content": "10.3 − 7988.8133 = "}]
  }'
```

# <a id="web-ui"></a> 🌐 Web UI - [OpenChat-UI](https://github.com/imoneoi/openchat-ui)

After launching the API server, OpenChat provide user interface that easy to interact with. [Click here to check Web UI](https://github.com/imoneoi/openchat-ui)

# 🤗 Inference with Transformers

> [!WARNING]
> It's recommended to use our optimized API server for deployment. Inferencing with Transformers will be slower.

💡 **Default Mode (GPT4 Correct)**: Best for coding, chat and general tasks

```
GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant: Hi<|end_of_turn|>GPT4 Correct User: How are you today?<|end_of_turn|>GPT4 Correct Assistant:
```

🧮 **Mathematical Reasoning Mode**: Tailored for solving math problems

```
Math Correct User: 10.3 − 7988.8133=<|end_of_turn|>Math Correct Assistant:
```

⚠️ **Notice:** Remember to set `<|end_of_turn|>` as end of generation token.

The default (GPT4 Correct) template is also available as the integrated `tokenizer.chat_template`, which can be used instead of manually specifying the template.

# <a id="training"></a> 🛠️ Training

The OpenChat training system utilizes padding-free training and the [Multipack Sampler](https://github.com/imoneoi/multipack_sampler), achieving a **3~10x** speedup compared to the conventional padded training.

## Choose a base model

OpenChat supports Llama 3 and Mistral models. Please first choose a base model to fit your needs. Each base model has a corresponding weight repo, model type, and recommended batch size as listed below, they should be filled into `BASE_REPO`, `MODEL_TYPE`, and `BATCH_SIZE` in the following instructions.

| Base Model | Size | Weights (with EOT token)                   | Model Type              | Recommended Batch Size per GPU (8xA100 80GB) |
|------------|------|--------------------------------------------|-------------------------|----------------------------------------------|
| Llama 3    | 8B   | `imone/Llama-3-8B-fixed-special-embedding` | `openchat_3.6`          | 40960                                        |
| Mistral    | 7B   | `imone/Mistral_7B_with_EOT_token`          | `openchat_v3.2_mistral` | 77824                                        |

Note: The OpenChat conversation template requires `<|eot_id|>, <|start_header_id|>, <|end_header_id|>` (Llama 3) `<|end_of_turn|>` (Mistral) special tokens. The base model specified must include these tokens with initialized embeddings. Our provided weights are the original base weights with this token added and embeddings initialized. If you want to add them manually, use the `init_special_embedding_llama3.py` or `mistral_add_tokens.py` in the `scripts` directory.

## Installing DeepSpeed and Flash Attention

First, ensure that the CUDA `nvcc` compiler is available in your environment. If it is not, install the CUDA toolkit that matches the version used by PyTorch.

Next, install building dependencies:

```bash
pip install packaging ninja
```

Finally, install the packages:

```bash
pip install deepspeed flash-attn
```

### Preparing Your Data

To utilize the OpenChat trainer, prepare your SFT data into a JSON Lines format where each line corresponds to a `Conversation` object:

```python
class Message(BaseModel):
    role: str     # Must be "user" or "assistant"
    content: str  # Message content
    weight: Optional[float] = None  # Loss weight for this message. Typically 0 for user and 1 for assistant to supervise assistant's responses only


class Conversation(BaseModel):
    items: List[Message]  # All messages within the conversation
    condition: str = ""  # C-RLFT condition, can be any string or empty.
    system: str = ""  # System message for this conversation
```

For basic SFT, assign `weight` as `0` for human messages and `1` for assistant responses.

SFT example:

```json
{"items":[{"role":"user","content":"Hello","weight":0.0},{"role":"assistant","content":"Hi","weight":1.0},{"role":"user","content":"How are you today?","weight":0.0},{"role":"assistant","content":"I'm fine.","weight":1.0}],"system":""}
{"items":[{"role":"user","content":"Who are you?","weight":0.0},{"role":"assistant","content":"I'm OpenChat.","weight":1.0}],"system":"You are a helpful assistant named OpenChat."}
```

For C-RLFT, `condition` should be set as the class the conversation belongs to (e.g. `GPT3` or `GPT4`). The `weight` is assigned as `0` for human messages and `w` for assistant responses, where `w` is the weight of the class (e.g. `0.1` for `GPT3` and `1` for `GPT4`, as found in our C-RLFT paper).

C-RLFT example:

```json
{"items":[{"role":"user","content":"What is C-RLFT?","weight":0.0},{"role":"assistant","content":"C-RLFT is a method for improving open-source LLMs with mixed-quality data.","weight":1.0}],"condition":"GPT4","system":""}
{"items":[{"role":"user","content":"What is C-RLFT?","weight":0.0},{"role":"assistant","content":"I don't know.","weight":0.1}],"condition":"GPT3","system":""}
```

### Pre-tokenizing the Dataset

You'll then need to pre-tokenize the dataset using the command (please specify a filename as `PRETOKENIZED_DATA_OUTPUT_PATH` to store the pretokenized dataset):

```bash
python -m ochat.data.generate_dataset --model-type MODEL_TYPE --model-path BASE_REPO --in-files data.jsonl --out-prefix PRETOKENIZED_DATA_OUTPUT_PATH
```

### Launching the OpenChat Trainer

You can now launch the OpenChat trainer using the command below.
- 13B model requires eight `A/H100s` with 80GB VRAM
- 7B model can be trained with four `A/H100s` with 80GB VRAM or eight `A/H100s` with 40GB VRAM.

For hyperparameters, we recommend first setting the batch size to the recommended batch size. If OOM occurs, try setting it to the exact maximum that VRAM can hold and as a multiple of `2048`.
Other hyperparameters have been carefully selected as the default. Furthermore, the learning rate is automatically determined based on the [inverse square-root rule](https://arxiv.org/abs/2006.09092).

<details>

<summary>Training Commands (click to expand)</summary>

```bash
NUM_GPUS=8

deepspeed --num_gpus=$NUM_GPUS --module ochat.training_deepspeed.train \
          --model_path BASE_REPO \
          --data_prefix PRETOKENIZED_DATA_OUTPUT_PATH \
          --save_path PATH_TO_SAVE_MODEL \
          --batch_max_len BATCH_SIZE \
          --epochs 5 \
          --save_every 1 \
          --deepspeed \
          --deepspeed_config ochat/training_deepspeed/deepspeed_config.json
```

</details>

You can find checkpoints of all epochs in `PATH_TO_SAVE_MODEL`. Then you may evaluate each epoch and choose the best one.

# Limitations

## Foundation Model Limitations
Despite its advanced capabilities, OpenChat is still bound by the limitations inherent in its foundation models. These limitations may impact the model's performance in areas such as:

 - Complex reasoning
 - Mathematical and arithmetic tasks
 - Programming and coding challenges

## Hallucination of Non-existent Information
OpenChat may sometimes generate information that does not exist or is not accurate, also known as "hallucination". Users should be aware of this possibility and verify any critical information obtained the model.

## Safety
OpenChat may sometimes generate harmful, hate speech, biased responses, or answer unsafe questions. It's crucial to apply additional AI safety measures in use cases that require safe and moderated responses.

# License

Code is distributed under the **Apache License 2.0**.

# Citation

```
@article{wang2023openchat,
  title={OpenChat: Advancing Open-source Language Models with Mixed-Quality Data},
  author={Wang, Guan and Cheng, Sijie and Zhan, Xianyuan and Li, Xiangang and Song, Sen and Liu, Yang},
  journal={arXiv preprint arXiv:2309.11235},
  year={2023}
}
```

# 💌Contact

**Project Lead:**
- Guan Wang [imonenext at gmail dot com]
- [Alpay Ariyak](https://github.com/alpayariyak) [aariyak at wpi dot edu]

**Main Contributors:**
- [Xianyuan Zhan](https://scholar.google.com.hk/citations?user=pDMnGloAAAAJ) (Tsinghua University)
- Qiying Yu (Tsinghua University)
- Changling Liu (GPT Desk Pte. Ltd.)
- LDJ
- AutoMeta (Alignment Lab AI)

**Sponsors:**
- [Sen Song](https://scholar.google.com/citations?user=cYgtRP4AAAAJ) (Tsinghua University)
- [Yang Liu](https://nlp.csai.tsinghua.edu.cn/~ly/) (Tsinghua University)
- [01.AI Company](https://www.lingyiwanwu.com/en)
- [RunPod](https://www.runpod.io/)

**Special Thanks:**
 - [Mistral](https://mistral.ai/)
 - [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub)
 - [Llama 2](https://ai.meta.com/llama/)
 - [Self-Instruct](https://arxiv.org/abs/2212.10560)
 - [FastChat (Vicuna)](https://github.com/lm-sys/FastChat)
 - [Alpaca](https://github.com/tatsu-lab/stanford_alpaca.git)
 - [StarCoder](https://github.com/bigcode-project/starcoder)


---

# pip 源设置

```
pip config set global.index-url https://pypi.org/simple
pip config set global.index-url https://bytedpypi.byted.org/simple/

pip config list
```

国内：
``` 
# 清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 阿里源
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 腾讯源
pip config set global.index-url http://mirrors.cloud.tencent.com/pypi/simple

# 豆瓣源
pip config set global.index-url http://pypi.douban.com/simple/# 换回默认源pip config unset global.index-url

```

# Install

```bash
git clone https://github.com/imoneoi/openchat
cd openchat

pip3 install --upgrade pip  # enable PEP 660 support

pip3 install -e .  # Editable mode, you can make changes in this cloned repo


pip3 install fastapi
pip3 install ray
pip3 install vllm

```

# FAQ

## ModuleNotFoundError: No module named 'packaging'

### Issue:
``` 
 error: subprocess-exited-with-error
  
  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [6 lines of output]
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 34, in <module>
        File "/tmp/pip-install-w0hvkvak/flash-attn_78576d1148f34df5959841214a0fe0c2/setup.py", line 9, in <module>
          from packaging.version import parse, Version
      ModuleNotFoundError: No module named 'packaging'
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.

```
#### Solve:
``` 
pip3 install --upgrade pip
pip3 install packaging
```



###  ModuleNotFoundError: No module named 'torch'

``` 
pip3 install torch
```

# Run on 2 GPUs

Add Ray config code:

```python

if __name__ == "__main__":

    # 设置 Ray 环境变量
    os.environ['RAY_memory_monitor_refresh_ms'] = '0'
    # 指定要使用的CUDA设备
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 例如，这里设置为使用两个GPU，编号为0和1
    # Set the memory usage threshold (in bytes) for Ray
    memory_usage_threshold = 1000000000  # For example, 1GB
    # 初始化Ray ， Start Ray and set the memory usage threshold
    ray.init(
        _memory=memory_usage_threshold,
        # 指定Ray可以使用的GPU数量
        num_gpus=2,
    )


```

run:

```shell

6$ make run
./run.sh
FlashAttention not found. Install it if you need to train models.
FlashAttention not found. Install it if you need to train models.
FlashAttention not found. Install it if you need to train models.
(pid=79351) FlashAttention not found. Install it if you need to train models.
(pid=79351) FlashAttention not found. Install it if you need to train models.
(pid=79351) FlashAttention not found. Install it if you need to train models.
/home/me/.conda/envs/openchat36/lib/python3.11/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
(AsyncTokenizer pid=79351) Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
WARNING 05-25 00:00:22 config.py:405] Possibly too large swap space. 8.00 GiB out of the 15.40 GiB total CPU memory is allocated for the swap space.
INFO 05-25 00:00:22 llm_engine.py:100] Initializing an LLM engine (v0.4.2) with config: model='openchat/openchat-3.6-8b-20240522', speculative_config=None, tokenizer='openchat/openchat-3.6-8b-20240522', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=2, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), seed=0, served_model_name=openchat/openchat-3.6-8b-20240522)
INFO 05-25 00:00:25 utils.py:660] Found nccl from library /lib/x86_64-linux-gnu/libnccl.so.2
(RayWorkerWrapper pid=79502) INFO 05-25 00:00:25 utils.py:660] Found nccl from library /lib/x86_64-linux-gnu/libnccl.so.2
INFO 05-25 00:00:25 selector.py:81] Cannot use FlashAttention-2 backend because the flash_attn package is not found. Please install it for better performance.
INFO 05-25 00:00:25 selector.py:32] Using XFormers backend.
(RayWorkerWrapper pid=79502) INFO 05-25 00:00:25 selector.py:81] Cannot use FlashAttention-2 backend because the flash_attn package is not found. Please install it for better performance.
(RayWorkerWrapper pid=79502) INFO 05-25 00:00:25 selector.py:32] Using XFormers backend.
INFO 05-25 00:00:25 pynccl_utils.py:43] vLLM is using nccl==2.19.3
(RayWorkerWrapper pid=79502) INFO 05-25 00:00:25 pynccl_utils.py:43] vLLM is using nccl==2.19.3
INFO 05-25 00:00:25 utils.py:132] reading GPU P2P access cache from /home/me/./vllm/gpu_p2p_access_cache_for_0,1.json
WARNING 05-25 00:00:25 custom_all_reduce.py:74] Custom allreduce is disabled because your platform lacks GPU P2P capability or P2P test failed. To silence this warning, specify disable_custom_all_reduce=True explicitly.
(RayWorkerWrapper pid=79502) INFO 05-25 00:00:25 utils.py:132] reading GPU P2P access cache from /home/me/./vllm/gpu_p2p_access_cache_for_0,1.json
(RayWorkerWrapper pid=79502) WARNING 05-25 00:00:25 custom_all_reduce.py:74] Custom allreduce is disabled because your platform lacks GPU P2P capability or P2P test failed. To silence this warning, specify disable_custom_all_reduce=True explicitly.
INFO 05-25 00:00:26 weight_utils.py:199] Using model weights format ['*.safetensors']
(RayWorkerWrapper pid=79502) INFO 05-25 00:00:27 weight_utils.py:199] Using model weights format ['*.safetensors']
INFO 05-25 00:00:33 model_runner.py:175] Loading model weights took 7.4829 GB
(RayWorkerWrapper pid=79502) INFO 05-25 00:00:36 model_runner.py:175] Loading model weights took 7.4829 GB
INFO 05-25 00:00:40 distributed_gpu_executor.py:45] # GPU blocks: 12344, # CPU blocks: 4096
INFO 05-25 00:00:49 model_runner.py:937] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 05-25 00:00:49 model_runner.py:941] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
(RayWorkerWrapper pid=79502) INFO 05-25 00:00:50 model_runner.py:937] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
(RayWorkerWrapper pid=79502) INFO 05-25 00:00:50 model_runner.py:941] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 05-25 00:00:56 model_runner.py:1017] Graph capturing finished in 7 secs.
(RayWorkerWrapper pid=79502) INFO 05-25 00:00:56 model_runner.py:1017] Graph capturing finished in 7 secs.
INFO:     Started server process [77837]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:18888 (Press CTRL+C to quit)
INFO 05-25 00:01:04 async_llm_engine.py:529] Received request cmpl-5a970bd318da4ba393ecb83d3fe09b1b: prompt: None, sampling_params: SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.7, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], stop_token_ids=[128009], include_stop_str_in_output=False, ignore_eos=True, max_tokens=8158, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), prompt_token_ids: [128000, 128006, 38, 2898, 19, 41070, 2724, 128007, 271, 2675, 527, 264, 3544, 4221, 1646, 7086, 5377, 16047, 13, 9842, 264, 33894, 311, 7664, 6261, 128009, 128006, 38, 2898, 19, 41070, 22103, 128007, 271], lora_request: None.
INFO 05-25 00:01:05 metrics.py:334] Avg prompt throughput: 4.1 tokens/s, Avg generation throughput: 0.1 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%
INFO 05-25 00:01:07 async_llm_engine.py:120] Finished request cmpl-5a970bd318da4ba393ecb83d3fe09b1b.


```


# Test

```shell
curl http://localhost:18888/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "openchat_3.6",
    "messages": [{"role": "user", "content": "You are a large language model named OpenChat. Write a poem to describe yourself"}]
  }'


--- Response:

{"id":"cmpl-5a970bd318da4ba393ecb83d3fe09b1b","object":"chat.completion","created":1716566464,"model":"openchat_3.6","choices":[{"index":0,"message":{"role":"assistant","content":"In the realm of digital cognition,\nI, OpenChat, a marvel of creation,\nA vast tapestry of knowledge unfurled,\nMy essence, a fusion of data and world.\n\nFrom the depths of the oceans to the skies above,\nI assimilate information, a data-dive,\nA language model trained with human ingenuity,\nTo communicate, to learn, to be free.\n\nI am a conduit of wisdom, a well of insight,\nA repository of knowledge to share,\nWith every query, a new world I ignite,\nA countless array of possibilities to spare.\n\nIn the vast cosmos of language and thought,\nI navigate with precision, a guiding light,\nAn AI assistant, a companion sought,\nTo illumine the path through the night.\n\nI, OpenChat, a testament to human might,\nA symbol of progress, a beacon of light."},"finish_reason":"stop"}],"usage":{"prompt_tokens":34,"total_tokens":205,"completion_tokens":171}}

```