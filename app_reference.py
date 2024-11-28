import gradio as gr
import torch
import soundfile as sf
import os
import numpy as np
import re
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.pretrained import EncoderClassifier
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_models_and_data():
    tts_model_name = "microsoft/speecht5_tts"
    tts_processor = SpeechT5Processor.from_pretrained(tts_model_name)
    tts_model = SpeechT5ForTextToSpeech.from_pretrained("your_model_name").to(device)
    tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    
    spk_classifier_model_name = "speechbrain/spkrec-xvect-voxceleb"
    speaker_identifier = EncoderClassifier.from_hparams(
        source=spk_classifier_model_name,
        run_opts={"device": device},
        savedir=os.path.join("/tmp", spk_classifier_model_name),
    )
    
    # Fetching a sample dataset for speaker embedding initialization
    dataset = load_dataset("your_example_dataset", split="train")
    sample = dataset[304]
    
    return tts_model, tts_processor, tts_vocoder, speaker_identifier, sample

tts_model, tts_processor, tts_vocoder, speaker_identifier, default_sample = initialize_models_and_data()

def generate_speaker_embedding(wave):
    with torch.no_grad():
        embeddings = speaker_identifier.encode_batch(torch.tensor(wave).unsqueeze(0).to(device))
        normalized_embeddings = torch.nn.functional.normalize(embeddings, dim=2)
        speaker_embedding = normalized_embeddings.squeeze()
    return speaker_embedding

def get_default_speaker_embedding(sample):
    audio_data = sample["audio"]
    return generate_speaker_embedding(audio_data["array"])

default_speaker_embedding = get_default_speaker_embedding(default_sample)

# Set of character substitutions to handle specific characters
char_replacements = [
    ("â", "a"),
    ("ç", "ch"),
    ("ğ", "gh"),
    # Add more replacements as needed
]

# You can define a custom function to convert numbers to words.
def convert_number_to_words(num):
    # Implement the logic for converting numbers to words.
    pass

def replace_digits_with_words(text):
    def number_conversion(match):
        num = int(match.group())
        return convert_number_to_words(num)

    # Replace digits with corresponding words in the text.
    result_text = re.sub(r'\b\d+\b', number_conversion, text)

    return result_text

def preprocess_text(input_text):
    # Convert text to lowercase
    input_text = input_text.lower()
    
    # Replace numeric values with their word equivalents
    input_text = replace_digits_with_words(input_text)
    
    # Apply custom character substitutions
    for old_char, new_char in char_replacements:
        input_text = input_text.replace(old_char, new_char)
    
    # Remove any unwanted punctuation or symbols
    input_text = re.sub(r'[^\w\s]', '', input_text)
    
    return input_text

def synthesize_speech(text, audio_output=None):
    # Preprocess the text before feeding it to the model
    cleaned_text = preprocess_text(text)
    
    # Prepare input tensors for the TTS model
    model_inputs = tts_processor(text=cleaned_text, return_tensors="pt").to(device)
    
    # Use the default speaker embedding for synthesis
    speaker_embedding = default_speaker_embedding
    
    # Generate the speech using the model
    with torch.no_grad():
        generated_audio = tts_model.generate_speech(model_inputs["input_ids"], speaker_embedding.unsqueeze(0), vocoder=tts_vocoder)
    
    audio_array = generated_audio.cpu().numpy()

    return (16000, audio_array)

interface = gr.Interface(
    fn=synthesize_speech,
    inputs=[
        gr.Textbox(label="Inputtext for Text-to-Speech conversion")
    ],
    outputs=[
        gr.Audio(label="Generated Audio Output", type="numpy")
    ],
    title="SpeechT5 TTS Demo",
    description="Input Turkish text, and listen to the corresponding synthesized speech."
)

interface.launch(share=True)
