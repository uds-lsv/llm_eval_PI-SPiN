

#Step 0: set the execution environment
source /data/users/achingacham/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_1_6_clone  #pytorch_1_6


#Step 1: paraphrase generation
#Inputs: A TAB-delimited file with input-sentences; column_name_or_num of input-sntences
#Outputs: Input file
#         +++ additional rows for output-paraphrases
#         +++ an additional column for 'paraphrase_pair_id' or 'parahrases_id'
# ---------------------------------------------------------------------------------
#  UPDATE the following variables with correct values before execution
#

input_file_path="data/300_input_sentences.tsv"
col_n=2

#ZSL-low
python3 API_paraphrase.py -iFile $input_file_path -iCol $col_n -uPrompt "Generate an intelligible paraphrase for the following input sentence: "

#ZSL-med
python3 API_paraphrase.py -iFile $input_file_path -iCol $col_n -uPrompt "Generate a simple, intelligible, spoken-styled paraphrase with 10 to 12 words for the following input sentence: "

#ZSL-high
python3 API_paraphrase.py -iFile $input_file_path -iCol $col_n -uPrompt "For a noisy listening environment with babble noise at SNR -5, generate a simple, intelligible, and spoken-styled paraphrase with 10-12 words, for the following input sentence:"

#PAS-n6
python3 API_paraphrase.py -iFile $input_file_path -iCol $col_n -uPrompt "Generate 6 simple, intelligible, spoken-styled paraphrases with 10 to 12 words for the following input sentence: "

#PAS-n12
python3 API_paraphrase.py -iFile $input_file_path -iCol $col_n -uPrompt "Generate 12 simple, intelligible, spoken-styled paraphrases with 10 to 12 words for the following input sentence: "

#ICL
