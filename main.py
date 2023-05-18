# ML
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
# torch specifically
import torch
from torch.optim import AdamW

# dataset utils
from datasets import load_from_disk

# Data
import scipy.signal as sps

# Ops
import os
import wandb
import pickle
from tqdm import tqdm
import random

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# weights and biases
hyperparametre_defaults = dict(
    lr = 3e-5,
    batch_size = 4,
    epochs = 32,
    data = "./data/SBCSAE_TURNS",
    model="openai/whisper-small"
)

# start wandb
wandb.init(project='chat-whisper', entity='jemoka', config=hyperparametre_defaults, mode="disabled")
# wandb.init(project='chat-whisper', entity='jemoka', config=hyperparametre_defaults)

# get config
config = wandb.config

DATA = config.data
BATCH_SIZE = config.batch_size
LR = config.lr
EPOCHS = config.epochs
MODEL = config.model
VAL_SAMPLES = 4

class ChatAudioData(Dataset):

    def __init__(self, datafile, sample_rate=44100):
        # load raw data
        self.raw_data = load_from_disk(datafile)

    def __getitem__(self, indx):
        # get sample
        data = self.raw_data[indx]

        return data["text"], data["audio"]

    def __len__(self):
        return len(self.raw_data)

# dataset
dataset = ChatAudioData(DATA)
train, val = torch.utils.data.random_split(
    dataset, [len(dataset)-VAL_SAMPLES, VAL_SAMPLES],
    generator=torch.Generator().manual_seed(1))

# train val split
dataloader = DataLoader(train, batch_size=BATCH_SIZE, collate_fn=lambda x:list(zip(*x)))
val_data = next(iter(DataLoader(val, batch_size=VAL_SAMPLES, collate_fn=lambda x:list(zip(*x)))))

# feature extractors
processor = WhisperFeatureExtractor.from_pretrained(MODEL, language="English", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained(MODEL, language="English", task="transcribe")

# model!
model = WhisperForConditionalGeneration.from_pretrained(f"{MODEL}").to(DEVICE)
optim = AdamW(model.parameters(), lr=LR)

# function to run validation
def run_log_val():
    text, audio = val_data

    # encode data
    encoded_audio = processor(audio, sampling_rate=16000, return_tensors="pt",
                              return_attention_mask=True, truncation=True,
                              max_length=30*16000).to(DEVICE)
    encoded_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True,
                             max_length=448).to(DEVICE)

    # pass through model
    out = model(input_features = encoded_audio["input_features"],
                attention_mask = encoded_audio["attention_mask"],
                labels=encoded_text["input_ids"])

    loss = out["loss"]

    id = random.randint(0,3)
    actual_out = tokenizer.batch_decode(torch.argmax(out["logits"], dim=2),
                                        skip_special_tokens=True)[id]
    expected_out = text[id]
    table = wandb.Table(columns=["output", "expected"])
    table.add_data(actual_out, expected_out)

    wandb.log({
        "val_sample": table,
        "val_loss": loss.detach().cpu().item()
    })

for e in range(EPOCHS): 
    print(f"Training epoch {e}...")
    for i, (text, audio) in enumerate(tqdm(iter(dataloader), total=len(dataloader))):

        # encode data
        encoded_audio = processor(audio, sampling_rate=16000, return_attention_mask=True, 
                                  return_tensors="pt", truncation=True, max_length=30*16000).to(DEVICE)
        encoded_text = tokenizer(text, return_tensors="pt",max_length=448,
                                 padding=True, truncation=True).to(DEVICE)

        # pass through model
        out = model(input_features = encoded_audio["input_features"],
                    attention_mask = encoded_audio["attention_mask"],
                    labels=encoded_text["input_ids"])

        loss = out["loss"]

        # optimization step
        loss.backward()
        optim.step()
        optim.zero_grad()

        # logging
        wandb.log({
            "train_loss": loss.detach().cpu().item()
        })

        # log example
        if i % 500 == 0:
            run_log_val()

# write model down
print("Saving model...")
os.mkdir(f"./models/{wandb.run.name}")
model.save_pretrained(f"./models/{wandb.run.name}")
tokenizer.save_pretrained(f"./models/{wandb.run.name}")


