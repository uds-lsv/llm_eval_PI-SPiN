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

### FEW-SHOT PROMPT ###
#best samples based on STOI at SNR -5
user_prompt=$"Look at the list of examples of a sentence and its intelligible paraphrase:
1. I don't know if you are familiar with that.  =>  I have no idea if you're familiar with that.
2. what other long range goals do you have besides college ?  =>  Apart from college ,  what are your other long - term objectives ?
3. I don't have access either. Although ,  I did at one time  =>  In the past ,  I had access ,  but currently ,  I don't.
4. Right now I've got it narrowed down to the top four teams. =>  At this point ,  I've trimmed my options and picked four top teams.
5. prohibition didn't stop it and didn't do anything really.  =>  It continued despite the prohibition ,  which didn't accomplish anything.
Similarly, generate an intelligible paraphrase  for the sentence:
"

# best samples based on Sent-Int (SNR 0)
#user_prompt=$"Look at the list of examples of a sentence and its intelligible paraphrase:
#1. Apart from college, what are your other long - term objectives ?  =>  what other long range goals do you have besides college ?
#2. Are you acquainted with that ? I'm not aware.  =>  I don't know you're familiar with that or not.
#3. Feeling stuck between two generations can be quite stressful.  =>  you feel squeezed in the middle of having both generations ,
#4. What is the typical timeframe for your contribution to become fully vested ?  =>  how long does it take for your contribution to vest ?
#5. In the past , I had access , but currently , I don't.  =>  I don't have access either. Although , I did at one time
#Similarly, generate an intelligible paraphrase  for the sentence:
#"

### ZERO-SHOT PROMPT ###
#p_e
user_prompt="Generate 6 simple, intelligible, spoken-styled paraphrases with 10 to 12 words for the following input sentence: "
#p_b
#user_prompt="Generate a simple, intelligible, spoken-styled paraphrase with 10 to 12 words for the following input sentence: "
#p_c
#user_prompt="For a noisy listening environment with babble noise at SNR -5, generate a simple, intelligible, and spoken-styled paraphrase with 10-12 words, for the following input sentence: "
#p_a
#user_prompt="Generate an intelligible paraphrase for the following input sentence: "
#p_f
#user_prompt="Generate 12 simple, intelligible, spoken-styled paraphrases with 10 to 12 words for the following input sentence: "


echo "### Input Prompt:  $user_prompt"

conda activate pytorch_1_6_clone;
pg_script="/nethome/achingacham/PycharmProjects/chatGPT/scripts/API_paraphrase.py"

python3 $pg_script \
         -iFile "$input_file_path" \
         -iCol "$col_n" \
         -uPrompt "$user_prompt" \
         -hRow 0 # -uPrompt

# LLaMA
# conda activate llama; pg_script="/nethome/achingacham/PycharmProjects/LLaMA/scripts/inference_llama.py.py"

# user_prompt="Generate a numbered list of 6 simple, intelligible, spoken-styled paraphrases with 10 to 12 words for the following input sentence: "
# python $pg_script \
#        -iFile $input_file_path \
#        -hRow 0 \
#        -iCol 'input_text' \
#        -uPrompt "$user_prompt"
