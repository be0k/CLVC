import os
import kaldi_io
import numpy as np
from tqdm import tqdm

mode = "timit"
splits = ['train', 'dev','test']
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



# import os
# import kaldi_io
# import numpy as np
# from tqdm import tqdm

# pdf2phone = {}
# with open("/home/main/Desktop/kaldi/egs/timit/s5/exp/tri4_nnet_ppg/pdf2phoneid.txt", "r") as f:
#     for line in f:
#         pdf, phone = map(int, line.strip().split())
#         pdf2phone[pdf] = phone
# phone_count = max(pdf2phone.values()) + 1  # 총 phone 개수

# splits = ['train', 'dev', 'test']
# base_ppg_dir = "/home/main/Desktop/kaldi/egs/timit/s5/exp/tri4_nnet_ppg"
# output_root = "/home/main/Desktop/tmp_ppg"

# os.makedirs(output_root, exist_ok=True)

# for split in splits:
#     print(f"🔁 Processing split: {split}")
#     ark_path = os.path.join(base_ppg_dir, split, "ppg.ark")
#     output_dir = os.path.join(output_root, split)
#     os.makedirs(output_dir, exist_ok=True)

#     for utt_id, mat in tqdm(kaldi_io.read_mat_ark(ark_path)):
#         # mat.shape = (T, 1920)
#         T, D = mat.shape
#         phone_mat = np.zeros((T, phone_count), dtype=np.float32)

#         for pdf_id in range(D):
#             phone_id = pdf2phone.get(pdf_id)
#             if phone_id is not None:
#                 phone_mat[:, phone_id] += mat[:, pdf_id]

#         # normalize
#         phone_mat /= phone_mat.sum(axis=1, keepdims=True)

#         # 저장
#         np.save(os.path.join(output_dir, f"{utt_id}.npy"), phone_mat)
