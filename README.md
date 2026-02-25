# TCLNet

### A Hybrid Transformer–CNN Framework Leveraging Language Models as Lossless Compressors for CSI Feedback
<img width="1593" height="786" alt="image" src="https://github.com/user-attachments/assets/3492324d-b767-4d66-9e57-dcd589887849" />
<img width="1559" height="785" alt="image" src="https://github.com/user-attachments/assets/d4f3c771-10ff-4ce8-847a-babe7bff2850" />


> [!CAUTION]
**Partial Code Release**
> 
> 
> This repository contains the **core modules** of TCLNet. The **complete implementation** (full training pipeline, evaluation scripts, and required intermediate symbol streams) will be released **after the paper is officially published**.
> 

---

## 📖 Overview

This repository provides the core implementation of **TCLNet**, a unified CSI feedback compression framework for FDD massive MIMO systems. TCLNet consists of:

1. **Lossy Compressor**: A hybrid Transformer–CNN architecture that captures both local spatial structures and long-range dependencies.
2. **Lossless Compressor**: An entropy coding framework combining a context-aware Language Model (LM), a Factorized Model (FM), a symbol selection mechanism, and arithmetic coding.

The framework is designed to balance the **Rate–Distortion–Complexity (RDC)** trade-off.

---

## 📦 What is Included

- **Lossy Compressor Modules**: Transformer–CNN hybrid blocks, attention blocks, TCBlock, and TransConv modules.
- **Lossless Compressor Core**: Core implementation of the entropy coding framework.

---

## 📂 Repository Structure

```
.
├── lossy/             # Lossy compressor modules (partial)
├── lossless/          # Lossless compressor core code (partial)
└── README.md          # Project documentation
```

> [!NOTE]
Folder names may differ depending on your local organization.
> 

---

## 🛠 Environment & Installation

### Requirements

We recommend using [Conda](https://docs.conda.io/en/latest/) to manage your environment.

| Dependency | Min. Version | Purpose |
| --- | --- | --- |
| **Python** | 3.9 | Base environment |
| **PyTorch** | 1.13 | Deep learning framework |
| **timm** | 0.6.12 | Transformer backbones |
| **compressai** | 1.2.4 | Entropy modeling & arithmetic coding |
| **einops** | 0.6.0 | Tensor manipulation |

### Quick Start

```bash
# 1. Clone the repository
git clone [<https://github.com/Zijiuyang/TCLNet.git>](<https://github.com/Zijiuyang/TCLNet.git>)
cd TCLNet

# 2. Create and activate a virtual environment
conda create -n tclnet python=3.9 -y
conda activate tclnet

# 3. Install core dependencies
pip install torch==1.13.1+cu117 torchvision --extra-index-url [<https://download.pytorch.org/whl/cu117>](<https://download.pytorch.org/whl/cu117>)
pip install timm einops compressai
```

---

## 🚀 Usage (Stay Tuned)

This repository is currently a **Partial Release**.

### 1. Data Preparation

Data preprocessing scripts for CSI datasets will be provided here once the paper is officially published.

### 2. Evaluation

It will be available after the paper is officially published.

---

## 📑 Citation

If you use this work in your research, please cite:

```
@article{yang2026tclnet,
  title   = {TCLNet: A Hybrid Transformer--CNN Framework Leveraging Language Models as Lossless Compressors for CSI Feedback},
  author  = {Yang, Zijiu and Yang, Qianqian and Tang, Shunpu and Yang, Tingting and Shi, Zhiguo},
  journal = {arXiv preprint arXiv:2601.06588},
  year    = {2026},
  url     = {[<https://arxiv.org/abs/2601.06588>](<https://arxiv.org/abs/2601.06588>)}
}
```

## 🙏 Acknowledgements

This repository is built upon and inspired by the following excellent open-source projects:

- [Python_CsiNet](https://github.com/sydney222/Python_CsiNet)
- [LIC_TCM](https://github.com/jmliu206/LIC_TCM)
- [language_modeling_is_compression](https://github.com/google-deepmind/language_modeling_is_compression)

We sincerely thank the authors for making their code publicly available.
