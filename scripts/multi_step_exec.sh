#!/bin/bash

# a script to execute the following:
# 1. paraphrase generation
# 2. speech synthesis
# 3. additive noise mixing + STOI calculation
# 4. best hypothesis selection (best hypo ~ highest STOI)

## general rules ###
## column numbers start with 1.
## use double quites to all bash "argument" that has a space-delimited list.

#Step 0: set the execution environment

HOME="/nethome/achingacham/"
####IMPORTANT if runing on Cluster; rename CUDA devices
#source /nethome/achingacham/HTCondor_prep/scripts/setup.sh
# comment this for Step 4 / uncomment it only for step 1

source /data/users/achingacham/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_1_6_clone  #pytorch_1_6

working_dir="/nethome/achingacham/PycharmProjects/step1_step4_noise_modeling/Step1_PipelineExecution/data_paraphrases_in_noise"

####### ASSIGN INPUTS ######
# Step 1: paraphrase generation (eg: TaPaCo)
s1_input_dir=$1 #"/projects/SFB_A4/llama-2/llama-data/SWDA-PiSPIN/" #"/projects/SFB_A4/A4-IntelligibleParaphrasesinNoise/data/HE_300_600/" #"/projects/SFB_A4/Corpora/TaPaCo/data/"
s1_input_file=$2 #"test.tsv" #"SWDA_short_utterances_300_600.tsv"
s1_input_file_path="$s1_input_dir/$s1_input_file"
s1_input_text_column='clean_text'  #2 #input_text

# Step 2: speech synthesis
s2_INPUT_DIR="$s1_input_dir"
s2_INPUT_FILE="$s1_input_file$3"  #"$s1_input_file-08-23-2023,220353_split_cols" #"$s1_input_file-08-10-2023,004302_split_cols"  #"$s1_input_file-07-28-2023,032537_split_cols"
s2_text_columns="system_response" #"system_response_1 system_response_2 system_response_3 system_response_4 system_response_5 system_response_6 system_response_7 system_response_8 system_response_9 system_response_10 system_response_11 system_response_12" #"input_text system_response_1 system_response_2 system_response_3 system_response_4 system_response_5 system_response_6" # a space separated list of columns; if numbers, starts with 1; else column name; it refers to the input to TTS
s2_extra_columns="paraphrases_id" # a space separated list of columns (numbers/names) for saving in output. eg: 'paraphrases_id'

# Step 3. additive noise mixing and STOI calculation
s3_dir_path_suffix="$s2_INPUT_DIR/$s2_INPUT_FILE-TTS-"
s3_dir_identifiers="system_response" #"system_response_1 system_response_2 system_response_3 system_response_4 system_response_5 system_response_6 system_response_7 system_response_8 system_response_9 system_response_10 system_response_11 system_response_12" #"input_text system_response_1 system_response_2 system_response_3 system_response_4 system_response_5 system_response_6" #a space separated ids
s3_noise_file_path="/projects/SFB_A4/AudioRepo/noises_5/babble"
s3_SNRs="-5" #a space separated ids
s3_clean_utt_col_num=4  #column numbers begin with 1
s3_pp_id_col_num=2  #column numbers begin with 1

noise_type=`echo $s3_noise_file_path | rev | cut -d '/' -f 1 | rev`


# Step 4. best hypothesis selection (best hypo ~ highest STOI)
s4_input_dir="$s3_dir_path_suffix"
s4_file_ids=`for snr in $s3_SNRs; do echo $noise_type"_"$snr; done`
s4_comparison_cols="system_response" #"system_response_1 system_response_2 system_response_3 system_response_4 system_response_5 system_response_6 system_response_7 system_response_8 system_response_9 system_response_10 system_response_11 system_response_12"
#s4_comparison_cols="system_response_1 system_response_2 system_response_3 system_response_4 system_response_5 system_response_6"
s4_criteria_col='STOI'
s4_comp_files_path="$s4_input_dir*/noise_$noise_type/*with_STOI*"  # !important to customize with wildbard
s4_output_file_prefix="$s2_INPUT_DIR/$s2_INPUT_FILE" #s4_output_file_prefix="$s1_input_file_path"

####### EXECUTE SCRIPTS ######
## Step 1: paraphrase generation
#bash $working_dir/paraphrase_generation.sh \
# "$s1_input_file_path" "$s1_input_text_column"


echo "End of Step 1"

## Step 2: speech synthesis
# bash $working_dir/paraphrase_speech_synthesis.sh \
# "$s2_INPUT_DIR" "$s2_INPUT_FILE" \
# "$s2_text_columns" "$s2_extra_columns"
#echo -e "org-text\t$s2_extra_columns\ttext\tclean_utt_path" > "$s2_INPUT_DIR/$s2_INPUT_FILE-"text2speech_all.txt
#tail -n +2 "$s2_INPUT_DIR/$s2_INPUT_FILE-TTS-"*/text2speech.txt >> "$s2_INPUT_DIR/$s2_INPUT_FILE-"text2speech_all.txt
#sed -i "/==>/d" "$s2_INPUT_DIR/$s2_INPUT_FILE-"text2speech_all.txt  #  delete unneccary file path
#sed -i "/^$/d" "$s2_INPUT_DIR/$s2_INPUT_FILE-"text2speech_all.txt   #  delete blank lines
#echo "Check out: "$s2_INPUT_DIR/$s2_INPUT_FILE-"txt2speech_all.txt"

echo "End of Step 2"

## 3. additive noise mixing

#bash $working_dir/paraphrase_noise_mixing.sh \
#      "$s3_dir_path_suffix" \
#      "$s3_dir_identifiers" \
#      "$s3_noise_file_path" "$s3_SNRs" \
#      "$s3_clean_utt_col_num" \
#      "$s3_pp_id_col_num"

echo "End of Step 3"

## 4. best hypothesis selection
bash $working_dir/best_hypothesis_selection.sh \
    "$s4_input_dir" \
    "$s4_file_ids" \
    "$s4_comparison_cols" \
    "$s4_criteria_col" \
    "$s4_comp_files_path" \
    "$s4_output_file_prefix"

echo "End of Step 4"


# Extra steps: prepare a directory with a list of all stimuli audio [for human experiment]
# bash /nethome/achingacham/PycharmProjects/general_scripts/bashScripts/make_listening_exp_dir.sh \
# a_TAB_delimited_input_file \
# the_col_num_abs_filepath \
# an_output_dir

# Add an additional column for Noise-level
# n=0; echo "NoiseLevel" > SNR_$n.txt; for i in `seq 1 30`; do echo "SNR_$n" >> SNR_$n.txt; done

# Report correlation in best-hypothesis selection [to verify STOI-based ranking] ##
# list_input_files=`echo $s4_output_file_prefix*best_hypo.txt.idx`
# [Pearson correlation is incorrect] python3 get_correlation.py --list_input_files  "$list_input_files" --input_column 'best_STOI-idx'

# combine multiple files; ( to create one data file)
# python3 ~/PycharmProjects/LLaMA/scripts/combine_files.py \
# -iFiles "path/to/multiple/files;path/to/multiple/files;path/to/multiple/files"
