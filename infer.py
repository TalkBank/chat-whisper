from sklearn.utils.validation import _num_samples
from torchaudio import transforms as T
from torchaudio import load

from transformers import pipeline

from dataclasses import dataclass

from collections import defaultdict

import torch
from transformers import WhisperProcessor
from simple_diarizer.diarizer import Diarizer

from nltk import sent_tokenize

# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# PYTORCH_ENABLE_MPS_FALLBACK=1
# pretrained model path
PRETRAINED = "openai/whisper-small"
# FILE = "./data/test.wav"
FILE = "../talkbank-alignment/broken2/input/53.wav"

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

    def __init__(self, model, language="english", target_sample_rate=16000):
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            chunk_length_s=30,
            stride_length_s=(3, 3),
            device=DEVICE,
            return_timestamps="word",
        )
        processor = WhisperProcessor.from_pretrained(model)

        # force decoder IDs to create language
        self.__decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")

        # diarizer
        self.__diar = Diarizer(embed_model='xvec', cluster_method='sc')

        # save the target sample rate
        self.sample_rate = target_sample_rate

    def load(self, f, num_speakers):
        """Load an audio file for procesing.

        Parameters
        ----------
        f : str
            The audio .wav file name to process.
        num_speakers : int
            The number of speakers

        Returns
        -------
        Tuple[ASRAudioFile, List[dict]]
            Return processed audio file and speaker segments.
        """

        # function: load and resample audio
        audio_arr, rate = load(f)

        # resample if needed
        if rate != self.sample_rate:
            audio_arr = T.Resample(rate, self.sample_rate)(audio_arr)

        # transpose and mean
        resampled = torch.mean(audio_arr.transpose(0,1), dim=1)
        segments = self.__diar.diarize(f, num_speakers=num_speakers)

        # and return the audio file
        return ASRAudioFile(f, resampled, self.sample_rate), segments

    def __call__(self, data, segments):
        words = self.pipe(data.cpu().numpy(),
                          batch_size=8, 
                          generate_kwargs = {"temperature": 0.5,
                                             "repetition_penalty": 1.5,
                                             "forced_decoder_ids": self.__decoder_ids})["chunks"]

        # we now perform the sweep line algorithm to align the
        # segment timestamps against the words
        groups = []

        for word in words:
            groups.append({
                "type": "word",
                "start": word["timestamp"][0],
                "end": word["timestamp"][1],
                "payload": word["text"]
            })

        for segment in segments:
            groups.append({
                "type": "segment",
                "start": segment["start"],
                "end": segment["end"],
                "payload": segment["label"]
            })


        # sorting the output to perform sweep
        groups = list(sorted(groups, key=lambda x:x["start"]))

        # tally turns together
        turns = []
        current_speaker = 0
        current_turn = []

        current_segment = groups.pop(0)
        while len(groups) > 0:
            element = groups.pop(0)

            if element["type"] == "word":
                current_turn.append({
                    "type": "word",
                    "ts": element["start"],
                    "end_ts": element["end"],
                    "value": element["payload"],
                })
            elif element["type"] == "segment" and current_speaker != element["payload"]:
                turns.append({
                    "elements": current_turn,
                    "speaker": current_speaker
                })
                current_speaker = element["payload"],
                current_turn = []

        turns.append({
            "elements": current_turn,
            "speaker": current_speaker
        })

        return {
            "monologues": turns
        }


# e = ASREngine(PRETRAINED, "english")
# audio, segments = e.load(FILE, 2)
# result = e(audio.all(), segments)
# words = raw["chunks"]


