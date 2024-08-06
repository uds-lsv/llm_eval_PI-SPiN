#!/bin/bash

# a script to execute the following:
# 1. paraphrase generation


#Step 0: set the execution environment
source /data/users/achingacham/anaconda3/etc/profile.d/conda.sh

#Step 1: paraphrase generation
#Inputs: A TAB-delimited file with input-sentences; column_name_or_num of input-sntences
#Outputs: Input file
#         +++ additional rows for output-paraphrases
#         +++ an additional column for 'paraphrases_id'
# ---------------------------------------------------------------------------------
# UPDATE the following variables with correct values before execution


 input_file_path=$1
 col_n=$2 #column number starts with 1

 #! import set the API key before calling the API

### Prompt-and-Select ###

#pas_n=6
user_prompt="Generate 6 simple, intelligible, spoken-styled paraphrases with 10 to 12 words for the following input sentence: "

#pas_n=12
user_prompt="Generate 12 simple, intelligible, spoken-styled paraphrases with 10 to 12 words for the following input sentence: "

echo "### Input Prompt:  $user_prompt"

conda activate pytorch_1_6_clone;
pg_script="scripts/API_paraphrase.py"

python3 $pg_script \
         -iFile "$input_file_path" \
         -iCol "$col_n" \
         -uPrompt "$user_prompt" \
         -hRow 0 # -uPrompt

