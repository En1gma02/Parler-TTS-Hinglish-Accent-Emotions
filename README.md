# Optimizing Multilingual Text-To-Speech with Accents and Emotions

Fine-tuned [Parler-TTS](https://github.com/huggingface/parler-tts) (600M) for Hinglish (Hindi-English code-mixed) language synthesis with Indian accent preservation and multi-scale emotion modelling. This work extends the Parler-TTS architecture with language-specific phoneme alignment, culture-sensitive emotion embedding layers trained on native speaker corpora, and dynamic accent code-switching with residual vector quantization.

**425+ downloads and 7 likes across all open-sourced models and datasets on Hugging Face. The research paper has received 26 upvotes on Hugging Face.**

[[Research Paper (arXiv)]](https://arxiv.org/abs/2506.16310) | [[Paper (Hugging Face)]](https://huggingface.co/papers/2506.16310) | [[Models]](https://huggingface.co/En1gma02) | [[Datasets]](https://huggingface.co/En1gma02)

---

## Table of Contents

- [Abstract](#abstract)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Inference -- Indian Accent (Hinglish)](#inference----indian-accent-hinglish)
  - [Inference -- English with Emotions](#inference----english-with-emotions)
- [Open-Sourced Models](#open-sourced-models)
- [Open-Sourced Datasets](#open-sourced-datasets)
- [Pipeline Overview](#pipeline-overview)
  - [Data Preparation](#data-preparation)
  - [Dataset Preprocessing](#dataset-preprocessing)
  - [Fine-Tuning](#fine-tuning)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)

---

## Abstract

State-of-the-art text-to-speech (TTS) systems realize high naturalness in monolingual environments, but synthesizing speech with correct multilingual accents (especially for Indic languages) and context-relevant emotions still poses difficulty owing to cultural nuance discrepancies in current frameworks. This work introduces a new TTS architecture integrating accent along with preserving transliteration with multi-scale emotion modelling, particularly tuned for Hindi and Indian English accent. The approach extends the Parler-TTS model by integrating a language-specific phoneme alignment hybrid encoder-decoder architecture, culture-sensitive emotion embedding layers trained on native speaker corpora, and dynamic accent code-switching with residual vector quantization. Subjective evaluation with 200 users reported a Mean Opinion Score (MOS) of 4.2/5 for cultural correctness, significantly outperforming existing multilingual systems (p<0.01).

---

## Key Results

| Metric | Value |
|---|---|
| Word Error Rate (WER) | 11.8% (23.7% reduction from baseline 15.4%) |
| Emotion Recognition Accuracy | 85.3% from native listeners |
| Mean Opinion Score (MOS) | 4.2 / 5.0 for cultural correctness |
| Statistical Significance | p < 0.01 vs. existing multilingual systems |
| Baselines Surpassed | METTS, VECL-TTS |

---

## Project Structure

```
Parler-TTS-Hinglish-Accent-Emotions/
|
|-- notebooks/
|   |-- Parler_TTS_Data_Preperation.ipynb         # Raw data collection and preparation
|   |-- HF_Parler_TTS_Dataset_Preprocessing.ipynb  # Dataset preprocessing and Hugging Face upload
|   |-- Finetuning_Parler_TTS_on_a_single_speaker_dataset.ipynb  # Model fine-tuning pipeline
|   |-- Use_Parler_TTS_Hinglish.ipynb              # Inference for Hinglish / Indian accent models
|   |-- Use_Parler_TTS_Emotions.ipynb              # Inference for emotion-conditioned models
|
|-- Research Paper.pdf          # Full research paper
|-- requirements.txt            # Python dependencies
|-- .gitignore
|-- README.md
```

---

## Installation

```bash
git clone https://github.com/En1gma02/Parler-TTS-Hinglish-Accent-Emotions.git
cd Parler-TTS-Hinglish-Accent-Emotions

pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for both fine-tuning and inference.

---

## Usage

### Inference -- Indian Accent (Hinglish)

```python
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained(
    "En1gma02/Parler-TTS-Mini-v0.1-Indian-Male-Accent-Hindi-Kaggle"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(
    "En1gma02/Parler-TTS-Mini-v0.1-Indian-Male-Accent-Hindi-Kaggle"
)

prompt = "Namaste, aaj hum baat karenge artificial intelligence ke baare mein."
description = "A male speaker with an Indian accent delivers the speech clearly and naturally."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()

sf.write("hinglish_output.wav", audio_arr, model.config.sampling_rate)
```

### Inference -- English with Emotions

```python
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained(
    "En1gma02/Parler-TTS-Mini-v1-English-Emotions"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(
    "En1gma02/Parler-TTS-Mini-v1-English-Emotions"
)

prompt = "I am so happy to see you after all these years!"
description = "A female speaker expresses happiness with a warm and excited tone."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()

sf.write("emotion_output.wav", audio_arr, model.config.sampling_rate)
```

For detailed usage with all models, refer to the notebooks in the `notebooks/` directory.

---

## Open-Sourced Models

All fine-tuned models are publicly available on Hugging Face:

| Model | Parameters | Downloads | Likes | Link |
|---|---|---|---|---|
| Parler-TTS-Mini-v0.1-Indian-Male-Accent-Hindi-Kaggle | 0.6B | 9 | 2 | [Link](https://huggingface.co/En1gma02/Parler-TTS-Mini-v0.1-Indian-Male-Accent-Hindi-Kaggle) |
| Parler-TTS-Mini-v1-English-Emotions | 0.9B | 7 | -- | [Link](https://huggingface.co/En1gma02/Parler-TTS-Mini-v1-English-Emotions) |
| Parler-TTS-Mini-v0.1-Indian-Accent-Kaggle | 0.6B | 3 | -- | [Link](https://huggingface.co/En1gma02/Parler-TTS-Mini-v0.1-Indian-Accent-Kaggle) |
| Parler-TTS-mini-v0.1-Indian-Accent | 0.6B | 2 | -- | [Link](https://huggingface.co/En1gma02/Parler_TTS_mini_v0.1_Indian_Accent) |

---

## Open-Sourced Datasets

All datasets used in this work are publicly available on Hugging Face:

| Dataset | Rows | Downloads | Likes | Link |
|---|---|---|---|---|
| hindi_speech_male_5hr | 5.84k | 132 | -- | [Link](https://huggingface.co/datasets/En1gma02/hindi_speech_male_5hr) |
| processed_hindi_speech_male_5hr | 5.84k | 63 | 1 | [Link](https://huggingface.co/datasets/En1gma02/processed_hindi_speech_male_5hr) |
| hindi_speech_female_5hr | 5.98k | 48 | 1 | [Link](https://huggingface.co/datasets/En1gma02/hindi_speech_female_5hr) |
| indian_accent_english_tagged | 6.77k | 38 | -- | [Link](https://huggingface.co/datasets/En1gma02/indian_accent_english_tagged) |
| processed_english_emotions | 2.91k | 34 | 2 | [Link](https://huggingface.co/datasets/En1gma02/processed_english_emotions) |
| indian_accent_english | 6.77k | 20 | 1 | [Link](https://huggingface.co/datasets/En1gma02/indian_accent_english) |
| processed_english_emotions_tagged | 2.91k | 17 | -- | [Link](https://huggingface.co/datasets/En1gma02/processed_english_emotions_tagged) |
| hindi_speech_10h | 11.8k | 15 | -- | [Link](https://huggingface.co/datasets/En1gma02/hindi_speech_10h) |
| hindi_speech_male_5hr_tagged | 5.84k | 11 | -- | [Link](https://huggingface.co/datasets/En1gma02/hindi_speech_male_5hr_tagged) |
| english_emotions | 2.91k | 10 | -- | [Link](https://huggingface.co/datasets/En1gma02/english_emotions) |
| processed_indian_accent_english | 6.77k | 9 | -- | [Link](https://huggingface.co/datasets/En1gma02/processed_indian_accent_english) |
| english_emotions_tagged | 2.91k | 7 | -- | [Link](https://huggingface.co/datasets/En1gma02/english_emotions_tagged) |

---

## Pipeline Overview

The end-to-end pipeline consists of three stages, each implemented as a standalone notebook.

### Data Preparation

**Notebook:** `notebooks/Parler_TTS_Data_Preperation.ipynb`

- Collects and curates raw Hindi speech data, Indian-accented English speech, and emotionally expressive English speech from multiple sources.
- Performs audio segmentation, normalization, and quality filtering.
- Prepares text transcriptions aligned with audio segments.

### Dataset Preprocessing

**Notebook:** `notebooks/HF_Parler_TTS_Dataset_Preprocessing.ipynb`

- Converts raw audio-text pairs into the format required by Parler-TTS.
- Generates speaker description tags (accent, gender, emotion, speaking style) for description-conditioned generation.
- Uploads processed datasets to Hugging Face Hub.

### Fine-Tuning

**Notebook:** `notebooks/Finetuning_Parler_TTS_on_a_single_speaker_dataset.ipynb`

- Fine-tunes the Parler-TTS Mini (600M / 900M) model on curated datasets.
- Configured for single-speaker fine-tuning with accent and emotion conditioning.
- Supports training on Google Colab (T4 GPU) and Kaggle environments.
- Trained models are pushed to Hugging Face Hub.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{pawar2025optimizing,
  title={Optimizing Multilingual Text-To-Speech with Accents \& Emotions},
  author={Pawar, Pranav and Dwivedi, Akshansh and Boricha, Jenish and Gohil, Himanshu and Dubey, Aditya},
  journal={arXiv preprint arXiv:2506.16310},
  year={2025},
  doi={10.48550/arXiv.2506.16310}
}
```

---

## Authors

- [Pranav Pawar](https://arxiv.org/search/cs?searchtype=author&query=Pawar,+P)
- [Akshansh Dwivedi](https://huggingface.co/En1gma02)
- [Jenish Boricha](https://arxiv.org/search/cs?searchtype=author&query=Boricha,+J)
- [Himanshu Gohil](https://arxiv.org/search/cs?searchtype=author&query=Gohil,+H)
- [Aditya Dubey](https://arxiv.org/search/cs?searchtype=author&query=Dubey,+A)

---

## License

This project is open-sourced for research purposes. Please refer to the [Parler-TTS license](https://github.com/huggingface/parler-tts/blob/main/LICENSE) for the base model terms.
