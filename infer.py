from torchaudio import transforms as T
from torchaudio import load

from transformers import pipeline

from dataclasses import dataclass

import torch
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer

from nltk import sent_tokenize

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')
# pretrained model path
PRETRAINED = "./models/smart-river-11"
# FILE = "./data/test.wav"
FILE = "../talkbank-alignment/tmp/input/minga01a.wav"

@dataclass
class ASRAudioFile:
    file : str
    tensor : torch.Tensor
    rate : int

    def chunk(self,begin_ms, end_ms):
        """Get a chunk of the audio.

        Parameters
        ----------
        begin_ms : int
            Milliseconds of the start of the slice.
        end_ms : int
            Milliseconds of the end of the slice.

        Returns
        -------
        torch.Tensor
            The returned chunk to supply to the ASR engine.
        """

        data = self.tensor[int(round((begin_ms/1000)*self.rate)):
                           int(round((end_ms/1000)*self.rate))]

        return data

    def all(self):
        """Get the audio in its entirety

        Notes
        -----
        like `chunk()` but all of the audio
        """

        return self.tensor

# inference engine
class ASREngine(object):
    """An ASR Engine

    Parameters
    ----------
    model : str
        The model path to load from.
    target_sample_rate : optional, int
        The sample rate to cast to. Defaults 16000 by Whisper.

    Example
    -------
    >>> engine = ASREngine("./model/my_model")
    >>> file = engine.load("./data/myfile.wav")
    >>> engine(file.chunk(7000, 13000)) # transcribes 7000th ms to 13000th ms
    """

    def __init__(self, model, target_sample_rate=16000):
        # load pretrained models
        # self.processor = WhisperFeatureExtractor.from_pretrained(model, language="English",
        #                                                          task="transcribe")
        # self.tokenizer = WhisperTokenizer.from_pretrained(model, language="English",
        #                                                   task="transcribe")
        # self.model = WhisperForConditionalGeneration.from_pretrained(model).to(DEVICE)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            chunk_length_s=18,
            stride_length_s=(3, 3),
            device=DEVICE,
            return_timestamps="word",
        )

        # save the target sample rate
        self.sample_rate = target_sample_rate

    def load(self, f):
        """Load an audio file for procesing.

        Parameters
        ----------
        f : str
            The audio .wav file name to process.

        Returns
        -------
        ASRAudioFile
            Return processed audio file.
        """

        # function: load and resample audio
        audio_arr, rate = load(f)

        # resample if needed
        if rate != self.sample_rate:
            audio_arr = T.Resample(rate, self.sample_rate)(audio_arr)

        # transpose and mean
        resampled = torch.mean(audio_arr.transpose(0,1), dim=1)

        # and return the audio file
        return ASRAudioFile(f, resampled, self.sample_rate)

    def __call__(self, data:torch.Tensor):
        # function: process audio file
        # encoded_audio = self.processor(data, sampling_rate=self.sample_rate,
        #                                return_attention_mask=True, return_tensors="pt",
        #                                truncation=True, max_length=30*self.sample_rate).to(DEVICE)

        # # call!
        # out = self.model.generate(input_features = encoded_audio["input_features"],
        #                           attention_mask = encoded_audio["attention_mask"],
        #                           max_new_tokens = 100000,
        #                           do_sample=True,
        #                           top_p=0.3).to(DEVICE)
        # # decode
        # decoded = self.tokenizer.decode(out[0].cpu(),
        #                                 skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        raw = self.pipe(data.cpu().numpy(),
                        batch_size=8, 
                        generate_kwargs = {"temperature": 0.5,
                                           "repetition_penalty": 1.5})

        # create draft utterances
        utt_drafts = []
        max_chunk = len(raw["chunks"])

        # group chunks together and create rough timestamps
        for i in range(0, max_chunk-1):
            # get the base text
            chunk = sent_tokenize(raw["chunks"][i]["text"])

            # there is space to go forward
            if i < max_chunk-1:
                chunk += [sent_tokenize(raw["chunks"][i+1]["text"])[0]]
            if i > 0:
                chunk += [sent_tokenize(raw["chunks"][i-1]["text"])[-1]]

            # check the first utterance; there's a chance that its already in
            # the previous utterance draft
            if len(utt_drafts) > 0 and utt_drafts[-1]["text"] == chunk[0]:
                chunk.pop(0)

            # append to the list!
            for c in chunk:
                # some cleanup
                c = c.strip()
                # last commas are not allowed
                if c[-1] == ",":
                    c = c[:-1]
                    c += '.'
                utt_drafts.append({"text": c, "timestamp": raw["chunks"][i]["timestamp"]})

        return utt_drafts


e = ASREngine(PRETRAINED)
audio = e.load(FILE)
results = e(audio.all())
