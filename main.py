from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="English", task="transcribe")

processor


