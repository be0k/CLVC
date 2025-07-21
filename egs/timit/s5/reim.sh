#!/usr/bin/env bash

#
# Copyright 2013 Bagher BabaAli,
#           2014-2017 Brno University of Technology (Author: Karel Vesely)
#
# TIMIT, description of the database:
# http://perso.limsi.fr/lamel/TIMIT_NISTIR4930.pdf
#
# Hon and Lee paper on TIMIT, 1988, introduces mapping to 48 training phonemes,
# then re-mapping to 39 phonemes for scoring:
# http://repository.cmu.edu/cgi/viewcontent.cgi?article=2768&context=compsci
#

. ./cmd.sh
[ -f path.sh ] && . ./path.sh
set -e

# Acoustic model parameters
numLeavesTri1=2500
numGaussTri1=15000
numLeavesMLLT=2500
numGaussMLLT=15000
numLeavesSAT=2500
numGaussSAT=15000
numGaussUBM=400
numLeavesSGMM=7000
numGaussSGMM=9000

feats_nj=10
train_nj=30
decode_nj=5

echo ============================================================================
echo "                Data & Lexicon & Language Preparation                     "
echo ============================================================================

#timit=/export/corpora5/LDC/LDC93S1/timit/TIMIT # @JHU
timit=/home/main/Desktop/kaldi/egs/timit/s5/data/TIMIT # @BUT

local/timit_data_prep.sh $timit || exit 1

local/timit_prepare_dict.sh

# Caution below: we remove optional silence by setting "--sil-prob 0.0",
# in TIMIT the silence appears also as a word in the dictionary and is scored.
utils/prepare_lang.sh --sil-prob 0.0 --position-dependent-phones false --num-sil-states 3 \
 data/local/dict "sil" data/local/lang_tmp data/lang

local/timit_format_data.sh

echo ============================================================================
echo "         MFCC Feature Extration & CMVN for Training and Test set          "
echo ============================================================================

# Now make MFCC features.
mfccdir=mfcc


for x in train dev test; do
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $feats_nj data/$x exp/make_mfcc/$x $mfccdir
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
done

echo ============================================================================
echo "                     MonoPhone Training & Decoding                        "
echo ============================================================================

steps/train_mono.sh  --nj "$train_nj" --cmd "$train_cmd" data/train data/lang exp/mono

utils/mkgraph.sh data/lang_test_bg exp/mono exp/mono/graph

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/mono/graph data/dev exp/mono/decode_dev

steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
 exp/mono/graph data/test exp/mono/decode_test


# # echo ============================================================================
# # echo "           tri1 : Deltas + Delta-Deltas Training & Decoding               "
# # echo ============================================================================

# # steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
# #  data/train data/lang exp/mono exp/mono_ali

# # # Train tri1, which is deltas + delta-deltas, on train data.
# # steps/train_deltas.sh --cmd "$train_cmd" \
# #  $numLeavesTri1 $numGaussTri1 data/train data/lang exp/mono_ali exp/tri1

# # utils/mkgraph.sh data/lang_test_bg exp/tri1 exp/tri1/graph

# # steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
# #  exp/tri1/graph data/dev exp/tri1/decode_dev

# # steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
# #  exp/tri1/graph data/test exp/tri1/decode_test

# # echo ============================================================================
# # echo "                 tri2 : LDA + MLLT Training & Decoding                    "
# # echo ============================================================================

# # steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
# #   data/train data/lang exp/tri1 exp/tri1_ali

# # steps/train_lda_mllt.sh --cmd "$train_cmd" \
# #  --splice-opts "--left-context=3 --right-context=3" \
# #  $numLeavesMLLT $numGaussMLLT data/train data/lang exp/tri1_ali exp/tri2

# # utils/mkgraph.sh data/lang_test_bg exp/tri2 exp/tri2/graph

# # steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
# #  exp/tri2/graph data/dev exp/tri2/decode_dev

# # steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
# #  exp/tri2/graph data/test exp/tri2/decode_test

# # echo ============================================================================
# # echo "              tri3 : LDA + MLLT + SAT Training & Decoding                 "
# # echo ============================================================================

# # # Align tri2 system with train data.
# # steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
# #  --use-graphs true data/train data/lang exp/tri2 exp/tri2_ali

# # # From tri2 system, train tri3 which is LDA + MLLT + SAT.
# # steps/train_sat.sh --cmd "$train_cmd" \
# #  $numLeavesSAT $numGaussSAT data/train data/lang exp/tri2_ali exp/tri3

# # utils/mkgraph.sh data/lang_test_bg exp/tri3 exp/tri3/graph

# # steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
# #  exp/tri3/graph data/dev exp/tri3/decode_dev

# # steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
# #  exp/tri3/graph data/test exp/tri3/decode_test

# # echo ============================================================================
# # echo "                        SGMM2 Training & Decoding                         "
# # echo ============================================================================

# # steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
# #  data/train data/lang exp/tri3 exp/tri3_ali

echo ============================================================================
echo "                    DNN Hybrid Training & Decoding                        "
echo ============================================================================

# DNN hybrid system training parameters
dnn_mem_reqs="--mem 1G"
dnn_extra_opts="--num_epochs 20 --num-epochs-extra 10 --add-layers-period 1 --shrink-interval 3"

steps/nnet2/train_tanh.sh --mix-up 5000 --initial-learning-rate 0.015 \
  --final-learning-rate 0.002 --num-hidden-layers 2  \
  --num-jobs-nnet "$train_nj" --cmd "$train_cmd" "${dnn_train_extra_opts[@]}" \
  data/train data/lang exp/mono exp/tri4_nnet

# Generate Docode Log Directory
[ ! -d exp/tri4_nnet/decode_dev ] && mkdir -p exp/tri4_nnet/decode_dev
decode_extra_opts=(--num-threads 6)

# Decoding dev (removing transform-dir)
steps/nnet2/decode.sh --cmd "$decode_cmd" --nj "$decode_nj" "${decode_extra_opts[@]}" \
  exp/mono/graph data/dev exp/tri4_nnet/decode_dev \
  | tee exp/tri4_nnet/decode_dev/decode.log

# Decoding test (removing transform-dir)
[ ! -d exp/tri4_nnet/decode_test ] && mkdir -p exp/tri4_nnet/decode_test
steps/nnet2/decode.sh --cmd "$decode_cmd" --nj "$decode_nj" "${decode_extra_opts[@]}" \
  exp/mono/graph data/test exp/tri4_nnet/decode_test \
  | tee exp/tri4_nnet/decode_test/decode.log


# echo ============================================================================
# echo "                              Extracting PPG                              "
# echo ============================================================================


model_dir=exp/tri4_nnet
out_base=exp/tri4_nnet_ppg
lang_dir=data/lang

# for split in train dev test; do
#   echo "=======> Processing [$split]"

#   data_dir=data/$split
#   feats_scp=$data_dir/feats.scp
#   out_dir=$out_base/$split

#   mkdir -p $out_dir

#   # 1. splice + LDA transform
#   echo "===> Step 1: Feature 준비 (splice + LDA transform) [$split]"
#   copy-feats scp:$feats_scp ark:- | \
#     splice-feats --left-context=3 --right-context=3 ark:- ark:- | \
#     transform-feats $model_dir/final.mat ark:- ark:$out_dir/feats.ark

#   # 2. forward pass: nnet2 softmax → PPG (1920 dim)
#   echo "===> Step 2: Extraction posterior to model forward [$split]"
#   nnet-am-compute \
#     --use-gpu=no \
#     $model_dir/final.mdl \
#     ark:$out_dir/feats.ark ark:$out_dir/ppg.ark
# done


for split in train dev test; do
  echo "=======> Processing [$split]"

  data_dir=data/$split
  out_dir=$out_base/$split
  mkdir -p $out_dir

  feats="ark,s,cs:apply-cmvn --utt2spk=ark:$data_dir/utt2spk scp:$data_dir/cmvn.scp scp:$data_dir/feats.scp ark:- |"

  echo "===> Step 2: Extraction posterior to model forward [$split]"
  nnet-am-compute --use-gpu=no \
    $model_dir/final.mdl "$feats" ark:$out_dir/ppg.ark
done


# Common Work : pdf - id <-> phone-id mapping (implementation once)
echo "===> Step 3: pdf-id → phone-id 매핑 파일 생성"
show-transitions $lang_dir/phones.txt $model_dir/final.mdl > $out_base/transitions.txt

show-transitions $lang_dir/phones.txt $model_dir/final.mdl | \
  grep "Transition-state" | \
  awk '{print $11, $5}' | sort | uniq > $out_base/pdf2phone.txt

awk '{print $1, $2}' $lang_dir/phones.txt > $out_base/phone2id.txt

join -1 2 -2 1 <(sort -k2 $out_base/pdf2phone.txt) <(sort $out_base/phone2id.txt) | \
  awk '{print $2, $3}' > $out_base/pdf2phoneid.txt

echo "Complete"
echo ============================================================================
echo "                    Getting Results [see RESULTS file]                    "
echo ============================================================================

bash RESULTS dev
bash RESULTS test


# echo ============================================================================
# echo "                                 I-vector                                 "
# echo ============================================================================


# steps/compute_vad_decision.sh --nj 4 --cmd "run.pl" data/train exp/vad_train
# for x in train dev test; do
#     steps/compute_vad_decision.sh --nj 4 --cmd "run.pl" data/$x exp/vad_$x
# done


# steps/train_diag_ubm.sh --nj 40 \
#     data/train 2048 \
#     exp/diag_ubm

# steps/train_full_ubm.sh --nj 4 --cmd "run.pl" data/train exp/diag_ubm exp/full_ubm


# # Train i-vector extractor based on UBM
# steps/train_ivector_extractor.sh --cmd "$train_cmd" \
#     --ivector-dim 150 --num-iters 5 \
#     exp/full_ubm/final.ubm data/train \
#     exp/extractor


# Extraction i-vector at each set.
# for x in train dev test; do
#     steps/extract_ivectors.sh --nj 4 exp/extractor data/$x exp/ivectors_$x
# done


# for x in train dev test; do
#     copy-vector ark:exp/ivectors_$x/spk_ivector.ark ark,t:exp/ivectors_$x/spk_ivector.txt

# done

echo ============================================================================
echo "Finished successfully on" `date`
echo ============================================================================

exit 0
