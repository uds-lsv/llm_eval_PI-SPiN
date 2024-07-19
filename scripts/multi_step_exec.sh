#!/bin/bash

# a script to execute the following:
# 1. paraphrase generation
# 2. speech synthesis
# 3. additive noise mixing
# 4. STOI calculation

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
# input_file_path="/projects/SFB_A4/Corpora/TaPaCo/data/TaPaCo_testset_P1_samples.txt"
# col_n=0
#
# python3 API_paraphrase.py -iFile $input_file_path -iCol $col_n -uPrompt "given the input sentence, generate 6 simple and intelligible paraphrases with less than 12 words:"


#Step 2: speech synthesis
#Inputs: ~ Output file from Step1; output_directory to save synthesized speech
#Outputs: Input file
#         +++ an additional column for 'clean_utt_path'
# ---------------------------------------------------------------------------------
# UPDATE the following variables with correct values before execution

#Step 3: additive noise mixing
#Inputs: Output file from Step2;  output_directory to save noisy speech; 'clean_utt_path' col_num; noise_file; SNR; 'paraphrase_pair_id' col_num;
#Outputs: Input file
#         +++ an additional column for 'noisy_utt_path'
# ---------------------------------------------------------------------------------
#  UPDATE the following variables with correct values before execution
#noiseMixingFile="/nethome/achingacham/sfba4_scripts/utils/create_mixed_audio_file.py"
#
#file_identifier=$1
#
#input_file=/projects/SFB_A4/TTS-in-Noise/PiN_synth_MultiSpk_VITS/metadata_"$file_identifier"_kan-bayashi-vctk_full_band_multi_spk_vits.csv  #"/projects/SFB_A4/TTS-in-Noise/PiN_synth/metadata_$file_identifier.csv"
#input_dir_path=/projects/SFB_A4/TTS-in-Noise/PiN_synth_MultiSpk_VITS/PiN_synth/kan-bayashi-vctk_full_band_multi_spk_vits_none_"$file_identifier"   #"/projects/SFB_A4/TTS-in-Noise/PiN_synth/$file_identifier" #'clean_utt_path'
#clean_utt_col_num=6  #column numbers begin with 1
#pp_id_col_num=5  #column numbers begin with 1
#
#noise_file_path="/projects/SFB_A4/AudioRepo/noise_1/babble"
#noise_type=`echo $noise_file_path | rev | cut -d '/' -f 1 | rev`
#SNR="-5"
#
#output_dir_path="$input_dir_path""_noisy"
#mkdir $output_dir_path
#
#output_dir_path="$output_dir_path/noise_$noise_type"
#mkdir $output_dir_path
#
#output_dir_path="$output_dir_path/snr_$SNR"
#mkdir $output_dir_path
#
#rm  /tmp/tmp_metadata.txt
#touch /tmp/tmp_metadata.txt
#
#
#count=0
#
#while read line ;
#do
#count=$((count+1))
#
#clean_file_path=`echo "$line" | cut -f "$clean_utt_col_num"`
#clean_file_path="$input_dir_path/$clean_file_path.wav"
#
#  pp_id=`echo "$line" | cut -f "$pp_id_col_num"`
#  utt_file_name=`echo "$clean_file_path" | rev | cut -d '/' -f 1 | rev`
#
#  #echo "$clean_file_path $noise_file_path $noise $SNR $utt_file_name"
#  ### add two columns 'clean_utt_path' and 'noisy_utt_path' if not exists in the metadata
#  echo -e "$clean_file_path\t$output_dir_path/$utt_file_name" >> /tmp/tmp_metadata.txt
#
#
#  # 3 possible methods to mix noise differs in the noise snippet selection.
#  # method 1: For noise mixing with a snippet that starts with a same index for all paraphrases
#  # python3 $noiseMixingFile -mix  --max_mix_length $pp_id --noise_file "$noise_file_path" --clean_file "$clean_file_path" --output_mixed_file "$output_dir_path/$utt_file_name" --snr $SNR  --output_clean_file "$input_dir_path/clean_after_NoiseMix/$utt_file_name"
#  #
#  # OR method 2: For noise mixing with a snippet that starts with a same index=1000
#  # python3 $noiseMixingFile -mix  --max_mix_length 1000 --noise_file "$noise_file_path" --clean_file "$clean_file_path" --output_mixed_file "$output_dir_path/$utt_file_name" --snr $SNR  --output_clean_file "$input_dir_path/clean_after_NoiseMix/$utt_file_name"
#  #
#  # OR method 3: For noise mixing with a random snippet of the noise file
#  # python3 $noiseMixingFile -mix --noise_file "$noise_file_path" --clean_file "$clean_file_path" --output_mixed_file "$output_dir_path/$utt_file_name" --snr $SNR  --output_clean_file "$input_dir_path/clean_after_NoiseMix/$utt_file_name"
#
#  #echo "Read input file $count lines"
#
#
#done < <(tail -n +2 $input_file) #to skip the first line
#
#echo -e "clean_utt_path\tnoisy_utt_path" > /tmp/tmp_header.txt
#cat /tmp/tmp_header.txt /tmp/tmp_metadata.txt > /tmp/tmp_utt_path.txt
#paste $input_file /tmp/tmp_utt_path.txt > $input_file"_for_STOI"


#Step 4: STOI calculation
#Inputs: Output file from Step3;
#Outputs: Input file
#         +++ an additional column for 'STOI'
# ---------------------------------------------------------------------------------
# UPDATE the following variables with correct values before execution

conda activate pytorch_1_6
python3 /nethome/achingacham/sfba4_scripts/utils/get_stoi.py -inFile $input_file"_for_STOI" -outFile "$output_dir_path/metadata_$file_identifier-with_STOI.txt"


