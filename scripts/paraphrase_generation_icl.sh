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
 col_n=$2 #the column of input text (col number starts with 1)

 #! import set the API key before calling the API

### In-context learning ### best samples based on STOI at SNR -5
#icl
user_prompt=$"Look at the list of examples of a sentence and its intelligible paraphrase:
1. I don't know if you are familiar with that.  =>  I have no idea if you're familiar with that.
2. what other long range goals do you have besides college ?  =>  Apart from college ,  what are your other long - term objectives ?
3. I don't have access either. Although ,  I did at one time  =>  In the past ,  I had access ,  but currently ,  I don't.
4. Right now I've got it narrowed down to the top four teams. =>  At this point ,  I've trimmed my options and picked four top teams.
5. prohibition didn't stop it and didn't do anything really.  =>  It continued despite the prohibition ,  which didn't accomplish anything.
Similarly, generate an intelligible paraphrase  for the sentence:
"


echo "### Input Prompt:  $user_prompt"

conda activate pytorch_1_6_clone;
pg_script="scripts/API_paraphrase.py"

python3 $pg_script \
         -iFile "$input_file_path" \
         -iCol "$col_n" \
         -uPrompt "$user_prompt" \
         -hRow 0 # -uPrompt

