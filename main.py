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

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# weights and biases
hyperparametre_defaults = dict(
    lr = 1e-5,
    batch_size = 16,
    epochs = 64,
    data = "./data/SBCSAE" 
)

# start wandb
wandb.init(project='chat-whisper', entity='jemoka', config=hyperparametre_defaults, mode="disabled")
# wandb.init(project='utok', entity='jemoka', config=hyperparametre_defaults)

# get config
config = wandb.config

DATA = config.data
BATCH_SIZE = config.batch_size
LR = config.lr
EPOCHS = config.epochs

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

dataset = ChatAudioData(DATA)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=lambda x:list(zip(*x)))
processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small", language="English", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en").to(DEVICE)
optim = AdamW(model.parameters(), lr=LR)

for _ in range(EPOCHS): 
    for i, (text, audio) in enumerate(tqdm(iter(dataloader), total=len(dataloader))):
        # encode data
        encoded_audio = processor(audio, sampling_rate=16000, return_tensors="pt")["input_features"].to(DEVICE)
        encoded_text = tokenizer(text, return_tensors="pt", padding=True).to(DEVICE)

        # pass through model
        out = model(input_features = encoded_audio,
                    attention_mask = encoded_text["attention_mask"],
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
        if i % 100 == 0:
            wandb.log({
                "sample": wandb.Html(tokenizer.batch_decode(torch.argmax(out["logits"], dim=2), skip_special_tokens=True)[0]),
                "target": wandb.Html(text[0])
            })


# write model down
print("Saving model...")
os.mkdir(f"./models/{wandb.run.name}")
model.save_pretrained(f"./models/{wandb.run.name}")
tokenizer.save_pretrained(f"./models/{wandb.run.name}")
procesor.save_pretrained(f"./models/{wandb.run.name}")


