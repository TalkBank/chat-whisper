# ML
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
# torch specifically
import torch
from torch.optim import AdamW
from accelerate import Accelerator, find_executable_batch_size

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

# deepspeed
from deepspeed.accelerator import get_accelerator

def execute():

    accelerator = Accelerator(log_with="wandb")
    DEVICE = accelerator.device

    # weights and biases
    config = dict(
        lr = 5e-6,
        batch_size = 1,
        epochs = 2,
        # epochs = 0,
        data = "./data/CWR",
        model="openai/whisper-large-v2",
        lora_alpha = 32,
        lora_dropout = 0.1,
        lora_target = ["q_proj", "v_proj", "out_proj"],
        r = 16,
    )

    # start wandb
    accelerator.init_trackers(project_name='chat-whisper',
                              init_kwargs={"wandb": {"entity": "jemoka"}},
                              config=config)

    peft_config = LoraConfig(inference_mode=False,
                             r=config["r"],
                             target_modules=config["lora_target"],
                             lora_alpha=config["lora_alpha"],
                             lora_dropout=config["lora_dropout"])
    # wandb.init(project='chat-whisper', entity='jemoka', config=hyperparametre_defaults)

    # get config
    DATA = config["data"]
    BATCH_SIZE = config["batch_size"]
    LR = config["lr"]
    EPOCHS = config["epochs"]
    MODEL = config["model"]
    VAL_SAMPLES = BATCH_SIZE

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
    model = WhisperForConditionalGeneration.from_pretrained(f"{MODEL}")
    model = get_peft_model(model, peft_config)
    model.train()

    # train only the decoder
    optim = AdamW(model.parameters(), lr=LR)

    # and 
    model, optim, dataloader, val_data = accelerator.prepare(model, optim, dataloader, val_data)

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

        loss = accelerator.gather(torch.mean(out["loss"]))
        logits = accelerator.gather(out["logits"])

        id = random.randint(0,BATCH_SIZE-1)
        actual_out = tokenizer.batch_decode(torch.argmax(logits, dim=2),
                                            skip_special_tokens=True)[id]
        expected_out = text[id]
        table = wandb.Table(columns=["output", "expected"])
        table.add_data(actual_out, expected_out)

        accelerator.log({
            "val_sample": table,
            "val_loss": torch.mean(loss).item()
        })

    for e in range(EPOCHS): 
        accelerator.print(f"Training epoch {e}...")
        for i, (text, audio) in enumerate(tqdm(iter(dataloader), total=len(dataloader))):
            # because large models
            get_accelerator().empty_cache()

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
            accelerator.backward(loss)
            optim.step()
            optim.zero_grad()

            loss = torch.mean(accelerator.gather(loss))

            # logging
            accelerator.log({
                "train_loss": loss.item()
            })

            # log example
            if i % 500 == 0:
                run_log_val()

    # write model down
    accelerator.end_training()
    accelerator.print("Saving model...")
    accelerator.wait_for_everyone()
    wandb_t = accelerator.get_tracker("wandb")
    os.mkdir(f"./models/{wandb_t.run.name}")
    accelerator.unwrap_model(model).merge_and_unload().save_pretrained(f"./models/{wandb_t.run.name}")
    tokenizer.save_pretrained(f"./models/{wandb_t.run.name}")
    processor.save_pretrained(f"./models/{wandb_t.run.name}")

if __name__ == "__main__":
    execute()
