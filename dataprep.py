import re
import os
import glob
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.io import wavfile

from datasets import load_dataset, Audio, Dataset

# sample rate
TARGET_SAMPLE_RATE=16000

# how many utterances to group into 1
GROUPS = 1

# input dirs
IN_DIR_TRANSCRIPTS = "./data/raw/CORALL/transcripts/"
IN_DIR_AUDIO = "./data/raw/CORALL/audio"
OUT_PATH = "./data/SBCSAE.parquet"

# get the actual files
in_files = sorted(glob.glob(os.path.join(IN_DIR_TRANSCRIPTS, "*.flo.cex")))

# load audio dataset
in_audios = load_dataset("audiofolder", data_dir=IN_DIR_AUDIO)
in_audios = in_audios.cast_column("audio", Audio(sampling_rate=16000))["train"]

def process_pair(f,w):
    """Processes a pair of files for data

    Parameters
    ----------
    f : str
        The input .flo.cex to process (output of `flo "+t*" "-t%mor" "-t%gra" "-t%wor"`)
    w : dict
        The dataset element to process, sample rate 16000

    Returns
    -------
    List[tuple]
        List containing elements of shape (text, timeslices (ms), audio sine)
    """

    with open(f, 'r') as df:
        text = df.read()
        # we want to get rid of lines that are just \n\t texttextetxet
        # because those are actually continuations of the previous line
        text = text.replace("\n\t", " ")
        # and now we get individual lines from the text for process
        # being careful to remove everything that is just a blank line
        # as an artifact from \n\t from above
        text = [i.strip() for i in text.split("\n") if i.strip() != ""]

    # filter out lines that are comments
    text = list(filter(lambda x:x[0] != "%", text))

    # filter out the speaker tags
    speakers = [re.search(r"\*(\w+):\t", i) for i in text]
    speakers = [i.group(1) if i else None for i in speakers]
    text = [re.sub(r"\*\w+:\t", "", i) for i in text]

    # now, we then grab the bulleted timeframes
    bullets = [re.search(r"\x15(\d+)_(\d+)\x15", i) for i in text]
    bullets = [(int(i.group(1)), int(i.group(2))) if i else None for i in bullets]

    # and then get rid of it from the bullets
    text = [re.sub(r"\x15(\d+)_(\d+)\x15", "", i).strip() for i in text]

    # seperate each turn out
    turns = []
    # prune one-off utterances
    turns_loc = list(filter(lambda x:x[1], enumerate(speakers)))
    # prune same speakers (to extract turns)
    # pruning rule: if the CURRENT speaker is the same as
    # the PREVIOUS speaker; prune it
    turns_loc = [x for i, x in enumerate(turns_loc)
                 if x[1] != turns_loc[i-1][1]]
    # create one-off pairwise shifts; cut the first one
    # which starts at "blank" and begins at time 0
    turns_index = list(zip([None]+turns_loc, turns_loc))[1:]
    turns_index = [(i[0], j[0]) for i,j in turns_index]
    # append the turn towards the end (i.e. the turn that ends
    # the file)
    turns_index.append((turns_index[-1][1], len(speakers)))

    # now, grab the bullets corresponding to the timepoints
    # one off error because the difference between slicing l[a:b] and indexing l[b-1]
    bullets_turns = [(bullets[i][0], bullets[j-1][1])
                     if (bullets[i] and bullets[j-1]) else None
                     for i,j in turns_index] 

    # now, audio
    mono_data = w["audio"]["array"]
    # and now, parcel out data alignments for each chunk
    # +500 smudge factor 
    mono_data_sliced = [mono_data[int(round((i[0]*TARGET_SAMPLE_RATE)/1000)):int(round((i[1]*TARGET_SAMPLE_RATE)/1000))]
                        if i else None for i in bullets_turns]

    # and text
    text_turns = [" ".join(text[i:j]) for i,j in turns_index]

    # now we roll
    text_time_tuples = list(zip(text_turns,
                                bullets_turns,
                                mono_data_sliced))

    # and filter out anything that isn't working
    def filter_result(result):
        return result[1] != None and result[2].shape[0] != 0
    text_time_tuples = list(filter(filter_result, text_time_tuples))

    # group signal into groups
    total = len(text_time_tuples)

    # get final results
    final_results = []
    for i in range(0, total, GROUPS):
        # cut into sections
        text, bullets, data = zip(*text_time_tuples[i:i+GROUPS])
        # combine text
        text = " ".join(text).strip()
        # get bullets
        bullets = [bullets[0][0], bullets[-1][-1]]
        # get data
        data = np.concatenate(data)

        final_results.append((text, bullets, data))

    # return!
    return final_results

# process!
results = []
for j in tqdm(in_audios):
    path = i["audio"]["path"]
    # calculate the transcript path
    i = os.path.join(IN_DIR_TRANSCRIPTS, f"{Path(path).stem}.flo.cex")
    # process!!
    results += process_pair(i, j)
    
# for i,j in tqdm(zip(in_files, in_audios), total=len(in_files)):

import pandas as pd
df = pd.DataFrame(results)
df.columns=["text", "timestamp", "audio"]

ds = Dataset.from_pandas(df)
ds_shuffled = ds.shuffle(seed=42)
ds_shuffled.save_to_disk("./data/CORALL_TURNS")


