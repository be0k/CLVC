import os
import kaldi_io
import numpy as np
from tqdm import tqdm

mode = "librispeech"
splits = ['dev_clean', 'dev_other', 'test_clean', 'test_other', 'train_clean_100']
base_ppg_dir = f"/home/main/Desktop/kaldi/egs/{mode}/s5/exp/tri4_nnet_ppg"
output_root = f"/home/main/Desktop/{mode}_ppg_raw"

os.makedirs(output_root, exist_ok=True)

for split in splits:
    print(f"🔁 Processing split: {split}")
    ark_path = os.path.join(base_ppg_dir, split, "ppg.ark")
    output_dir = os.path.join(output_root, split)
    os.makedirs(output_dir, exist_ok=True)

    for utt_id, mat in tqdm(kaldi_io.read_mat_ark(ark_path)):
        # mat.shape = (T, D) = raw PPG (pdf-level)
        np.save(os.path.join(output_dir, f"{utt_id}.npy"), mat)

