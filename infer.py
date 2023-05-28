from torchaudio import transforms as T
from torchaudio import load

import torch
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
TARGET_SAMPLE_RATE=16000

# pretrained model path
PRETRAINED = "./models/valiant-sponge-9"

# load pretrained models
processor = WhisperFeatureExtractor.from_pretrained(PRETRAINED, language="English",
                                                    task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained(PRETRAINED, language="English",
                                             task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(PRETRAINED)

# function: load and resample audio
f = "./data/test.wav"
audio_arr, rate = load(f)

# resample if needed
if rate != TARGET_SAMPLE_RATE:
    audio_arr = T.Resample(rate, TARGET_SAMPLE_RATE)(audio_arr)

# transpose and mean
resampled = torch.mean(audio_arr.transpose(0,1), dim=1)

# function: process audio file
encoded_audio = processor(resampled, sampling_rate=TARGET_SAMPLE_RATE,
                          return_attention_mask=True, return_tensors="pt",
                          truncation=True, max_length=30*TARGET_SAMPLE_RATE).to(DEVICE)

# call!
out = model.generate(input_features = encoded_audio["input_features"],
                     attention_mask = encoded_audio["attention_mask"],
                     max_new_tokens = 100000).to(DEVICE)
# decode
decoded = tokenizer.decode(out[0],
                           skip_special_tokens=True, clean_up_tokenization_spaces=True)


