import re
import os
import glob
import pickle
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile

# sample rate
SAMPLE_RATE=44100

# input dirs
IN_DIR_TRANSCRIPTS = "./data/raw/SBCSAE/transcripts/"
IN_DIR_AUDIO = "./data/raw/SBCSAE/audio/"
OUT_PATH = "./data/SBCSAE.pkl"

# get the actual files
in_files = sorted(glob.glob(os.path.join(IN_DIR_TRANSCRIPTS, "*.flo.cex")))
in_audios = sorted(glob.glob(os.path.join(IN_DIR_AUDIO, "*.wav")))

def process_pair(f,w):
    """Processes a pair of files for data

    Parameters
    ----------
    f : str
        The input .flo.cex to process (output of simply `flo "+t*" *`)
    w : str
        The .wav file to process, sample rate 44100

    Returns
    -------
    List[tuple]
        List containing elements of shape (text, timeslices (ms), audio sine)
    """

    with open(f, 'r') as df:
        text = [i.strip() for i in df.readlines()]

    # filter out lines that are comments
    text = list(filter(lambda x:x[0] != "%", text))

    # filter out the speaker tags
    text = [re.sub(r"\*\w+:\t", "", i) for i in text]

    # now, we then grab the bulleted timeframes
    bullets = [re.search(r"\x15(\d+)_(\d+)\x15", i) for i in text]
    bullets = [(int(i.group(1)), int(i.group(2))) if i else None for i in bullets]

    # and then get rid of it from the bullets
    text = [re.sub(r"\x15(\d+)_(\d+)\x15", "", i).strip() for i in text]

    # now, audio
    rate, data = wavfile.read(w)
    # extract left channel and take sin (NO IDEA WHY but apparently that's
    # the actual activations)
    try:
        mono_data = np.sin(data[:, 0])
    except IndexError:
        # our file is mono!
        mono_data = np.sin(data)
    # and now, parcel out data alignments for each chunk
    mono_data_sliced = [mono_data[int(round(i[0]/1000))*SAMPLE_RATE:int(round(i[1]//1000))*SAMPLE_RATE]
                        if i else None for i in bullets]

    # now we roll
    text_time_tuples = list(zip(text,bullets,mono_data_sliced))

    # and filter out anything that isn't working
    def filter_result(result):
        return result[1] != None and result[2].shape[0] != 0
    text_time_tuples = list(filter(filter_result, text_time_tuples))

    # return!
    return text_time_tuples

# process!
results = []
for i,j in tqdm(zip(in_files, in_audios), total=len(in_files)):
    results += process_pair(i, j)

# dump!
with open(OUT_PATH, "wb") as df:
    pickle.dump(results, df)
                

