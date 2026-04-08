# Fine-Tuning Qwen3 for Austrian Parliament Speech Classification

LoRA fine-tuning of a small LLM (Qwen3-0.6B) for two multi-class classification tasks on speeches from the Austrian Parliament: predicting the **political party** of the speaker, and predicting the **ideological orientation** of that party.

Created as a graded assignment for the course **"Deep Learning for Social Sciences"** (Summer Semester 2025) at the Department of Politics and Public Administration, University of Konstanz, taught by Dr. Giordano de Marzo.

## What this project does

- **Builds a clean dataset** from the [ParlaMint-AT corpus](https://www.clarin.si/repository/xmlui/handle/11356/1912), covering Austrian parliamentary speeches from 2018 to 2022. After filtering procedural speeches, removing duplicates, and dropping speeches with missing party labels, the working corpus contains 16,396 speeches across six parties.
- **Performs exploratory data analysis** including speech length distributions, party-level violin plots, temporal patterns, vocabulary statistics, and per-party word clouds with German stopword filtering.
- **Establishes zero-shot baselines** by prompting the base Qwen3-0.6B model with chat-template instructions to classify speeches.
- **Fine-tunes Qwen3-0.6B with LoRA** (Low-Rank Adaptation via the `peft` library) using instruction-formatted training data. Training is done in causal-LM mode with rank 16, scaling factor 32, and dropout 0.1, adding ~10M trainable parameters on top of the frozen base model.
- **Evaluates both base and fine-tuned models** on stratified test sets using accuracy, weighted F1, classification reports, and confusion matrices.

## Results

Fine-tuning produces a clear improvement on party prediction (F1 from 0.09 to 0.43) but barely moves the needle on ideological orientation. Overall performance is modest, mainly because the project was constrained to a 0.6B base model and a 30% training subset by Colab's GPU memory limits. The full results, confusion matrices, and a discussion of failure modes are in [`report.pdf`](report.pdf).

## Tech stack

`transformers` · `peft` (LoRA) · `datasets` · `torch` · `scikit-learn` · `pandas` · `seaborn` · `wordcloud` · `nltk`

## How to run

The notebook is designed to run end-to-end in Google Colab with a GPU runtime. The ParlaMint-AT corpus is downloaded directly from the CLARIN.SI repository on first run, and all dependencies install via pip. Open `qwen3_finetuning_parliament_classification.ipynb` in Colab and execute cells from top to bottom.

## Course context

- **Course**: Deep Learning for Social Sciences (Summer Semester 2025)
- **Department**: Politics and Public Administration, University of Konstanz
- **Instructor**: Dr. Giordano de Marzo
- **Student**: Richard Neureuther
