#!/usr/bin/env bash

HOME="/nethome/achingacham/"

#temporary comment
#source /data/users/achingacham/anaconda3/etc/profile.d/conda.sh
#conda activate pytorch_1_6

root_dir="/nethome/achingacham/PycharmProjects/para_metrics/para_metrics/"

### for SWDA sentence vs. ChatGPT best-paraphrases ###

input_file=$1 #"/projects/SFB_A4/A4-IntelligibleParaphrasesinNoise/data/HE_300_600/SWDA_short_utterances_300_600.tsv-08-24-2023,235121-babble_-5.txt-best_sent.txt_stimuli_all_with_trivial.txt"

#/projects/SFB_A4/A4-IntelligibleParaphrasesinNoise/data/HE_300_600//SWDA_short_utterances_300_600.tsv-09-13-2023,021544-babble_-5.txt-best_hypo.txt_stimuli_all_with_trivial.txt - p_{zsl-low}

echo "Inside get_para_metrics.sh"

python3 "$root_dir/get_para_metrics.py" \
        -iFH 0 -iF-sC "input_text" -iF-psC "best_hypo" \
        --input_file_include_cols "paraphrases_id,STOI-input_text,STOI-best_hypo,best_hypo_ratio_STOI" \
        -iF $input_file


        # -iF "/projects/SFB_A4/A4-IntelligibleParaphrasesinNoise/data/HE_300_600/SWDA_short_utterances_300_600.tsv-08-16-2023,214846-babble_-5.txt-best_hypo.txt_stimuli_all_wo_trivial.txt"
        # -iF "/projects/SFB_A4/A4-IntelligibleParaphrasesinNoise/data/HE_300_600/SWDA_short_utterances_300_600.tsv-08-16-2023,214846-babble_-5.txt-best_hypo.txt_stimuli_all.txt"


        #-iF "/projects/SFB_A4/A4-IntelligibleParaphrasesinNoise/data/HE_300_600/SWDA_short_utterances_300_600.tsv-08-11-2023,233716_split_cols-babble_-5.txt-best_hypo.txt_stimuli_random_30.txt"
        #-iF "/projects/SFB_A4/A4-IntelligibleParaphrasesinNoise/data/HE_300_600/SWDA_short_utterances_300_600.tsv-08-11-2023,233716_split_cols-babble_-5.txt-best_hypo.txt_stimuli_all_wo_trivial.txt"


        #-iF "/projects/SFB_A4/A4-IntelligibleParaphrasesinNoise/data/HE_300_600/SWDA_short_utterances_300_600.tsv-08-11-2023,233716_split_cols-babble_-5.txt-best_hypo.txt_stimuli_top_30.txt" \
        #-iF "/projects/SFB_A4/A4-IntelligibleParaphrasesinNoise/data/HE_300_600/SWDA_short_utterances_300_600.tsv-08-11-2023,233716_split_cols-babble_-5.txt-best_hypo.txt_stimuli_random_30.txt"


#finetuning with huggingface_
#echo "TaPaCo (test) with bart-base-all"; python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-all_2023_04_19_23:35:23/TaPaCo_testset_P1_nbests.txt" -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'
#echo "TaPaCo (test) with bart-base-reordered (old)"; python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-reorder_2023_04_20_00:09:41/TaPaCo_testset_P1_nbests.txt" -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'

#echo "TaPaCo (test) with bart-base-filtered (with pSTOI)"; python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-filter-pSTOI_2023_04_20_00:48:16/TaPaCo_testset_P1_nbests.txt" -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'
#echo "TaPaCo (test) with bart-base-filtered(pSTOI)_reordered"; python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-filter-pSTOI-reorder_2023_04_20_01:11:57/TaPaCo_testset_P1_nbests.txt" -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'

#echo "TaPaCo (test) with bart-base-filtered (with LD)"; python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-filter-LD_2023_04_20_01:32:24/TaPaCo_testset_P1_nbests.txt" -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'
#echo "TaPaCo (test) with bart-base-filtered(LD)_reordered"; python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-filter-LD-reorder_2023_04_20_01:53:59/TaPaCo_testset_P1_nbests.txt" -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'


# extras (April 28, 2023)
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-all-rs31_2023_04_28_02:27:24/TaPaCo_testset_P1_nbests.txt" -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'
#
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-all-rs42_MQP_2023_04_28_03:01:21/TaPaCo_testset_P1_nbests.txt" -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'
#
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-all-rs42_P_2023_04_28_03:38:25/TaPaCo_testset_P1_nbests.txt" -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'
#
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-all-rs53_2023_04_28_04:15:25/TaPaCo_testset_P1_nbests.txt"  -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'

#############

#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-reorder-rs31_2023_04_28_04:37:39/TaPaCo_testset_P1_nbests.txt" -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'

##python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-reorder-rs42_MQP_2023_05_02_19:21:46/TaPaCo_testset_P1_nbests.txt" -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'
# a re-run for reorder
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-reorder-rs42_MQP_2023_05_24_01:25:30/TaPaCo_testset_P1_nbests.txt" -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'

#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-reorder-rs42_P_2023_05_02_19:40:03/TaPaCo_testset_P1_nbests.txt"  -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'

#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-reorder-rs53_2023_04_28_05:00:38/TaPaCo_testset_P1_nbests.txt"  -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'

############

#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-filter1.03-rs42_MQP_2023_05_24_01:43:04/TaPaCo_testset_P1_nbests.txt"  -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'


# MQP_filter-reorder
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-reorder-filter1.02-rs42_MQP_2023_05_03_21:26:41/TaPaCo_testset_P1_nbests.txt"  -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-reorder-filter1.03-rs42_MQP_2023_05_03_21:44:22/TaPaCo_testset_P1_nbests.txt"  -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-reorder-filter1.04-rs42_MQP_2023_05_03_22:03:38/TaPaCo_testset_P1_nbests.txt"  -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'

# with length regularization:
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-reorder-filter1.03-reg1.0-rs42_MQP_2023_05_16_01:40:15/TaPaCo_testset_P1_nbests.txt"  -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-reorder-filter1.03-reg10-rs42_MQP_2023_05_16_02:13:20/TaPaCo_testset_P1_nbests.txt"  -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-reorder-filter1.03-reg0.1-rs42_MQP_2023_05_24_01:03:30/TaPaCo_testset_P1_nbests.txt" -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'


## with chatGPT output:
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/TaPaCo_testset_P1.txt-05-20-2023,235630" -iFH 0 -iF-sC 'input_text' -iF-psC 'system_response'
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/TaPaCo_testset_P1.txt-05-21-2023,001605" -iFH 0 -iF-sC 'input_text' -iF-psC 'system_response'
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/TaPaCo_testset_P1.txt-05-21-2023,004407" -iFH 0 -iF-sC 'input_text' -iF-psC 'system_response'
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/TaPaCo_testset_P1_chatGPT_audio/ts-05-21-2023,010444/best_STOI_Text_with_Source.tsv" -iFH 0 -iF-sC 'Source' -iF-psC 'best_STOI_Text'
# all generated hypothesis:
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/TaPaCo_testset_P1.txt-05-21-2023,010444_split_cols"  -iFH 0 -iF-sC 'input_text' -iF-psC 'system_response_1'
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/TaPaCo_testset_P1.txt-05-21-2023,010444_split_cols"  -iFH 0 -iF-sC 'input_text' -iF-psC 'system_response_2'
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/TaPaCo_testset_P1.txt-05-21-2023,010444_split_cols"  -iFH 0 -iF-sC 'input_text' -iF-psC 'system_response_3'
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/TaPaCo_testset_P1.txt-05-21-2023,010444_split_cols"  -iFH 0 -iF-sC 'input_text' -iF-psC 'system_response_4'
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/TaPaCo_testset_P1.txt-05-21-2023,010444_split_cols"  -iFH 0 -iF-sC 'input_text' -iF-psC 'system_response_5'
#best_STOI-hypo 
#python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/withSTOI-all.txt-best_hypo.txt"  -iFH 0 -iF-sC "Words-input_text" -iF-psC "best_STOI-Words"



# finetuning with fairseq_
#echo "TaPaCo (test) with bart-base-reordered"; python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-reordered/test.hypo_checkpoint_best.pt.tsv_nbests.tsv" -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'
#echo "TaPaCo (test) with bart-base-filtered (with pSTOI)"; python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-filtered/test.hypo_checkpoint_best.pt.tsv_nbests.tsv" -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'
#echo "TaPaCo (test) with bart-base-filtered(pSTOI)_reordered"; python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/TaPaCo/data/bart-base-filtered-pSTOI_reordered/test.hypo_checkpoint_best.pt.tsv_nbests.tsv" -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'

##echo "TaPaCo (test) with para-bart."; python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/para-metrics/ideal_paraphrases/MRPC_QQP_old/checkpoints-base-e50-metricloss/test.hypo_checkpoint_best.pt.tsv_nbests.tsv_bs1" -iFH 0 -iF-sC 'src_0' -iF-psC 'hyp_0'
##

## Different datasets:

#for datasplit in aa
#do
#echo "Reduced ParaBank1.0-small part $datasplit"; python3  "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/parabank-1.0-small-diverse/20230425-235639/reduce_dataset.tsv-$datasplit" -iF-sC 0 -iF-psC 1
#done

#for datasplit in ab ac ad ae af ag ah ai aj ak al am an ao ap aq ar as  #aa ab ac ad ae af ag ah ai aj ak al am an ao ap as : 'aq' & 'ar' are missing
#do
#echo "Reduced ParaBank2.0 top1 among 5 parahrases: $datasplit"; python3 "$root_dir/get_para_metrics.py" -iF "/projects/SFB_A4/Corpora/parabank-2.0/20230425-233609/reduce_dataset.tsv-$datasplit" -iF-sC 1 -iF-psC 2
#done


#echo "QQP 100 samples..."; python3 "$root_dir/get_para_metrics.py" -iF "../data/QQP_all_valid_paraphrases_100.txt" -iFH 0 -iF-sC '#1 Question' -iF-psC '#2 Question'
#
#echo "TaPaCo (test) ..."; python3 "$root_dir/get_para_metrics.py" -iF "../data/TaPaCo_CorpusSententialParaphrases_refined_testset.txt" -iFH 0 -iF-sC 'Sentence' -iF-pidC 'PS_ID'
#
#echo "MRPC ..."; python3 "$root_dir/get_para_metrics.py" -iF "../data/MRPC_all_valid_praphrases.txt" -iFH 0 -iF-sC '#1 String' -iF-psC '#2 String'
#
###echo "QQP ..."; python3 "$root_dir/get_para_metrics.py" -iF "../data/QQP_all_valid_paraphrases.txt" -iFH 0 -iF-sC '#1 Question' -iF-psC '#2 Question'
#
#echo "QQP with real STOI ..."; python3 "$root_dir/get_para_metrics.py" -iF "../data/QQP_test_with_all_feats.tsv" -iFH 0 -iF-sC 'Short Utterance' -iF-pidC 'paraphrase_pair_id' -iF-gtC 'STOI'
#
#echo "PAWS-WiKi ..."; python3 "$root_dir/get_para_metrics.py" -iF "../data/PAWS_WiKi_all_valid_paraphrases.txt" -iFH 0 -iF-sC 'sentence1' -iF-psC 'sentence2'
#
##TapaCo (all: this has several paraphrases in each set)


#For samples from chatGPT
#input_file='../data/results-ChatGPT.tsv'
#python3 "$root_dir/get_para_metrics.py" -iF "$input_file" -iFH 0 -iF-sC 'Input Sentence' -iF-psC 'Please generate a paraphrase that is half of the length of the original sentence:'

## For PiN data splits
#echo "PiN ..."; python3 "$root_dir/get_para_metrics.py" -iF "../data/PiN.tsv"   -iF-vCs "STOI,ppl,ph_len" -iFH 0 -iF-sC 'Short Utterance' -iF-pidC 'paraphrase_pair_id' -iF-gtC 'PhER'
#echo "PiN_both ..."; python3 "$root_dir/get_para_metrics.py" -iF "../data/PiN_both.tsv"  -iF-vCs "STOI,ppl,ph_len" -iFH 0 -iF-sC 'Short Utterance' -iF-pidC 'paraphrase_pair_id' -iF-gtC 'PhER'
#echo "PiN_either ..."; python3 "$root_dir/get_para_metrics.py" -iF "../data/PiN_either.tsv"  -iF-vCs "STOI,ppl,ph_len" -iFH 0 -iF-sC 'Short Utterance' -iF-pidC 'paraphrase_pair_id' -iF-gtC 'PhER'
##
## For PiN data splits (per SNR)
#echo "PiN_splits ..."; python3 "$root_dir/get_para_metrics.py" -iF "../data/PiN.tsv"  -pGC "Noise_ID" -iF-vCs "STOI,ppl,ph_len" -iFH 0 -iF-sC 'Short Utterance' -iF-pidC 'paraphrase_pair_id' -iF-gtC 'PhER'
#echo "PiN_both_splits ..."; python3 "$root_dir/get_para_metrics.py" -iF "../data/PiN_both.tsv" -pGC "Noise_ID" -iF-vCs "STOI,ppl,ph_len" -iFH 0 -iF-sC 'Short Utterance' -iF-pidC 'paraphrase_pair_id' -iF-gtC 'PhER'
#echo "PiN_either_splits ..."; python3 "$root_dir/get_para_metrics.py" -iF "../data/PiN_either.tsv" -pGC "Noise_ID"  -iF-vCs "STOI,ppl,ph_len" -iFH 0 -iF-sC 'Short Utterance' -iF-pidC 'paraphrase_pair_id' -iF-gtC 'PhER'

echo "Exiting get_para_metrics.sh"
