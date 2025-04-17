#!/bin/bash

# a script to execute the following:
# 4. best hypothesis selection (based on STOI)

#Step 0: set the execution environment
source path_to/conda.sh
conda activate pytorch_1_6_clone  #pytorch_1_6

#Step 4: best hypothesis (higher STOI) selection
#Inputs: ~
#Outputs: Input file
#         +++ an additional columns:
# ---------------------------------------------------------------------------------
# UPDATE the following variables with correct values before execution


script_dir="./"

input_dir=$1
file_ids=$2
comparison_cols=$3
criteria_col=$4

for file_id in $file_ids
do
  comp_files_path=$5
  output_file_prefix=$6

  date_ts=`date +"%F_%T"`
  output_file="$output_file_prefix-$file_id.txt"

  rm $output_file
  touch $output_file

  #if a single file with all hypotheses is not present create it first.
  #paste <(cut -f 16- FILE_1.txt) <(cut -f 16- FILE_2.txt) <(cut -f 16- FILE_3.txt) <(cut -f 16- FILE_4.txt) <(cut -f 16- FILE_5.txt) > OUTPUT_FILE.txt
  #input_dir="/projects/SFB_A4/Corpora/TaPaCo/data/TaPaCo_testset_P1_chatGPT_audio/gTTS-TaPaCo"  #"/projects/SFB_A4/Corpora/TaPaCo/data/"
  #output_file="$input_dir/withSTOI-all.txt"

  for file in `ls $comp_files_path | grep  -e "$file_id" `  #`ls $input_dir/*tsv`
  do
  #echo $file
  paste <(cut -f 1- "$output_file") <(cut -f 1- "$file") > "$output_file-temp"
  cp "$output_file-temp" "$output_file"

#  echo "#### No of headings in  $file -> $output_file ####"
#  head -n1 "$output_file-temp" | sed 's/\s/\n/g' | wc -l

  done

  sed -i 's/^\t//g' $output_file #remove the initial TAB
  rm "$output_file-temp"

  #input a file with both input and nbest hypothesis for selecting the best-hypothesis based on a criteria
  python3 $script_dir/select_best_hypo.py \
        -iF $output_file \
        -oF "$output_file-best_hypo.txt" \
        -cCols "$comparison_cols" \
        -criCol $criteria_col \
        -aPairCols "clean_utt_path,noisy_utt_path" \
        #-iFF '/projects/SFB_A4/A4-IntelligibleParaphrasesinNoise/data/HE_300_600/SWDA_short_utterances_300_600.tsv-08-11-2023,233716_split_cols-babble_-5.txt-best_hypo.txt_stimuli_random_30.txt'

  # Get para-metrics on the generated utterances:
  bash ./get_para_metrics.sh "$output_file-best_hypo.txt_stimuli_all_with_trivial.txt"

  # Get para-metrics on SELECTED generated utterances:
  bash ./get_para_metrics.sh "$output_file-best_hypo.txt_selected_stimuli.txt"

  # for files with an additional comparison step
#  python3 $script_dir/select_best_hypo.py \
#        -iF $output_file \
#        -oF "$output_file-best_sent.txt" \
#        -cCols "$comparison_cols" \
#        -criCol $criteria_col \
#        -aPairCols "clean_utt_path,noisy_utt_path" \
#        --compare_with_input  # comment this for all evaluation except post-processing with STOI
#
#  bash /nethome/achingacham/PycharmProjects/para_metrics/para_metrics/get_para_metrics.sh "$output_file-best_sent.txt_stimuli_all_with_trivial.txt"

echo "Check out the log files"
ls "$output_file"* | grep 'log'
done