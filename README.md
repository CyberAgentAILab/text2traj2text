<div align="center">
<h1>Text2Traj2Text: Learning-by-Synthesis Framework for Contextual Captioning of Human Movement Trajectories</h3>

<p align="center">
    <a href="https://hikaruasano.github.io/">Hikaru Asano</a><sup>1</sup> &nbsp;
    <a href="https://yonetaniryo.github.io/">Ryo Yonetani</a><sup>2</sup> &nbsp;
    <a href="https://scholar.google.co.jp/citations?user=7dLBoF0AAAAJ&hl=ja">Taiki Sekii</a><sup>2</sup> &nbsp;
    <a href="https://hiroki13.github.io/">Hiroki Ouchi</a><sup>2,3</sup>
</p>

<p align="center">
    <sup>1</sup>The University of Tokyo &nbsp;
    <sup>2</sup>CyberAgent &nbsp;
    <sup>3</sup>Nara Institute of Science and Technology
</p>

<p align="center">
    <strong>INLG 2024</strong>
</p>

<p align="center">
    <!-- <a href="https://arxiv.org/abs"><img src="https://img.shields.io/badge/arXiv-paper-orange" alt="arXiv paper"></a> -->
    <!-- <a href="https://udonda.github.io/RALF/"><img src="https://img.shields.io/badge/Project-Page-Green" alt="Project Page"></a> -->
    <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

</div>

---

## 📌 Overview

**Text2Traj2Text** is a learning-by-synthesis framework designed to generate natural language captions that describe the contextual backgrounds of shoppers' trajectory data in retail environments. The framework comprises two primary components:

1. **Text2Traj**: Generates customer behavior descriptions and corresponding trajectory data.
2. **Traj2Text**: Trains a model to convert trajectory data into natural language captions.

---

## 🛠 Prerequisites
We checked the reproducibility under the following environment.
- **Operating System**: Ubuntu (≥22.04) or macOS
- **Docker**: Version 24.0.7
- **Docker Compose**: Version 2.23.1
- **CUDA 11.8.0**: For GPU support
- **Python 3.9+**

---

## 🔧 Setup

We recommend using Docker to manage dependencies. Follow the steps below to set up the environment.

### 1. Install Docker

If you haven't installed Docker yet, please follow the [Docker Installation Guide](https://docs.docker.com/get-docker/).

### 2. Build and Run the Docker Container
Execute the following command to build and run the Docker container:

```bash
docker compose up -d
```

This command initializes a containerized environment with all necessary dependencies.

### 3. Preprocess Data

Before training, preprocess the data. Place the raw training data in the [`data`](./data/) directory and run:

```bash
bash scripts/preprocess.sh
```

---

## 🎓 Initial Experiment

To train and evaluate the **Traj2Text** model, execute the following command:

```bash
docker exec text2traj2text python3 scripts/train.py
```

By default, this command trains the model using the `t5-small` architecture with 8 paraphrased datasets.

---

## 🚀 Advanced: Full Reproduction of Our Experiment

To fully reproduce our experiments, you will need:

- Access to the **Azure OpenAI API** (for Text2Traj dataset generation)
- A GPU (for training models like `t5-base` and evaluating with LLaMA)

### Customizing Training Parameters

You can customize the training process by specifying parameters. For example, to train the model using `t5-base` with 0 paraphrased data points:

```bash
docker exec text2traj2text python3 scripts/train.py train.model_name=t5-base dataset.num_paraphrase=0
```

---

## ⚙️ Generating Text2Traj Dataset

### API Keys and Environment Variables

API keys are necessary for generating the Text2Traj dataset and evaluating models with LLaMA and OpenAI.

To generate the dataset or run evaluations, follow these steps:

1. Create a `.env` file in the root directory of the project.
2. Add the following content to the `.env` file:

   ```bash
   AZURE_OPENAI_VERSION=
   AZURE_OPENAI_ENDPOINT=
   AZURE_OPENAI_API_KEY=
   HUGGINGFACE_ACCESS_TOKEN=
   ```

   - The Azure OpenAI API key is required for Text2Traj dataset generation and evaluation with ChatGPT.
   - The Hugging Face access token is required for evaluation with LLaMA-2-7b.

To generate the **Text2Traj** dataset, follow these steps:

1. **Generate User Captions**:

   ```bash
   docker exec text2traj2text python3 scripts/text2traj/generate_user_captions.py
   ```

2. **Generate Purchase List**:

   ```bash
   docker exec text2traj2text python3 scripts/text2traj/generate_purchase_list.py
   ```

3. **Generate Paraphrasing**:

   ```bash
   docker exec text2traj2text python3 scripts/text2traj/generate_paraphrasing.py
   ```

4. **Generate Trajectory**:

   ```bash
   docker exec text2traj2text python3 scripts/text2traj/generate_trajectory.py
   ```

### Execute All Steps at Once

To run all the above steps sequentially, use:

```bash
bash scripts/generate_user_activity.sh
```

### Adjusting Generation Parameters

You can modify parameters such as `num_generations` and `model_name` directly in the script. For example, to generate 1000 data points using `gpt-4o` with a temperature of 0.7:

```bash
docker exec text2traj2text python3 scripts/text2traj/generate_user_captions.py num_generations=1000 model_name=gpt-4o temperature=0.7
```

### Saving Generated Datasets

Generated datasets are stored in the `data/raw_data/<project_name>` directory. To specify a different project name:

```bash
docker exec text2traj2text python3 scripts/text2traj/generate_user_captions.py project_name=your_project_name
```

### Training with the Generated Dataset

After dataset generation, preprocess it before training:

```bash
bash scripts/preprocess.sh your_project_name
```

---

## 📊 Evaluation with In-Context Learning

### 3.1 Evaluation with ChatGPT

To evaluate using GPT series models (e.g., GPT-4, GPT-3.5-turbo):

```bash
docker exec text2traj2text python3 scripts/eval_chatgpt.py
```

### 3.2 Evaluation with Open Source LLM

To evaluate using open-source language models (e.g., LLaMA-2-7b):

```bash
docker exec text2traj2text python3 scripts/eval_llm.py
```

---

## 📄 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{asano2024text2traj2text,
    title={{Text2Traj2Text: Learning-by-Synthesis Framework for Contextual Captioning of Human Movement Trajectories}},
    author={Hikaru Asano and Ryo Yonetani and Taiki Sekii and Hiroki Ouchi},
    booktitle={arXiv},
    year={2024}
}
```
