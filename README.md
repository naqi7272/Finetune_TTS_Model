# Fine-tuning Text-to-Speech (TTS) Models for English Technical Speech and Regional Languages

## Project Overview
This repository contains the solution for fine-tuning two Text-to-Speech (TTS) models. The project focuses on improving TTS model performance in two domains:
1. English technical speech (with a focus on technical jargon commonly used in interviews).
2. A regional language of your choice.

Additionally, optimization techniques such as quantization are explored to improve inference speed and reduce model size without significantly compromising audio quality.


## Objectives
The key objectives of this project are:
1. **Fine-tune a TTS model for English technical terms**: Enhance the model's pronunciation of commonly used technical terms like "API," "CUDA,".The finetuned model is available here on Hugging Face: https://huggingface.co/naqi72/speecht5_finetuned_techincal_data
 
2. **Fine-tune a TTS model for a regional language**: Improve the model's ability to synthesize high-quality, natural-sounding speech in a regional language.The finetuned model is available here on Hugging Face: https://huggingface.co/naqi72/speecht5_tts_voxpopuli_hindi

3. **Optimize model for fast inference**: Apply quantization techniques to speed up inference while maintaining output quality.

## Installation

**Clone the Repository:**

```
git clone https://github.com/naqi72/Finetuning_TTS_Model
cd Finetuning_TTS_Model
```
**Install Dependencies:** Install the required Python libraries from requirements.txt:
```
pip install -r requirements.txt
```

**Download Pre-Trained Model:** 
```
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor
model = SpeechT5ForTextToSpeech.from_pretrained('your-model-checkpoint')
processor = SpeechT5Processor.from_pretrained('your-model-checkpoint')
```

## Usage:

**Using in Your Python Script:** You can directly use the model in a Python script to convert text to speech:
```
import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor

# Load fine-tuned model and processor
model = SpeechT5ForTextToSpeech.from_pretrained('your-model-checkpoint')
processor = SpeechT5Processor.from_pretrained('your-model-checkpoint')

# Example text input
text_input = "Enter your text here."

# Preprocess text and convert to speech
inputs = processor(text_input, return_tensors="pt")
with torch.no_grad():
    speech = model.generate_speech(inputs["input_ids"])

# Save the output as a .wav file
torchaudio.save('output.wav', speech, 16000)
```

## Task Breakdown

### 1. English TTS Fine-Tuning
- **Model Selection**
- **Dataset**
- **Fine-Tuning**
- **Evaluation**
### 2. Regional Language TTS Fine-Tuning
- **Model Selection**
- **Dataset**
- **Fine-Tuning**
- **Evaluation**

### 3. Fast Inference Optimization (Optional)
- **Quantization Techniques**
- **Inference Speed**

## Deliverables
- Fine-tuned models for both English technical terms and the regional language.
- Quantized models (optional).
- Generated audio samples for both pre-trained and fine-tuned models.
- Performance reports and evaluation logs.
