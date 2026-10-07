# 论文代码实验环境

两套环境均位于项目目录内，使用 Python 3.10、PyTorch 2.5.1、torchvision 0.20.1 和 CUDA 12.1 运行时，互不混装。

## BC-LLM

```bash
conda activate /home/src/Reliable/.venvs/bc-llm
cd /home/src/Reliable/src/bc-llm
python scripts/train_bayesian.py --help
```

该环境包含 LangChain/OpenAI/Anthropic/Bedrock 客户端、Transformers、Datasets、Sentence Transformers、medspaCy、WILDS、SCons 和 NLTK `words` 语料。实际调用 GPT-4o 时仍需配置 API 凭据。

## C²BM

```bash
conda activate /home/src/Reliable/.venvs/c2bm
cd /home/src/Reliable/src/causally-reliable-cbm
python main.py --config-name sweep_c2bm_bn --cfg job
```

该环境包含 Lightning、Hydra/list-sweeper、causal-learn、CDT、bnlearn/pgmpy、PyG、LangChain、Mistral/OpenAI 客户端、Albumentations、W&B/Plotly/Kaleido，以及 NLTK `punkt`/`punkt_tab` 语料。`pgmpy` 固定为 `0.1.25`，避免 `bnlearn 0.10.2` 调用已从 pgmpy 1.0 移除的 `BayesianNetwork` 类。

激活环境时会自动移除系统 ROS 注入的 `PYTHONPATH`，并将 Matplotlib、Hugging Face、Torch 和 C²BM 缓存指向 `/home/src/Reliable/.cache`；退出环境时恢复原变量。

## 基础自检

```bash
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

精确安装版本见同目录下的 `bc-llm.freeze.txt` 和 `c2bm.freeze.txt`。
