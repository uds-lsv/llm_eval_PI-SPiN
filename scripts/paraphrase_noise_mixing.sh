#!/bin/bash

# a script to execute the following:
# 3. additive noise mixing (for a given noise file and SNR(s))

#Step 0: set the execution environment
source /data/users/achingacham/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_1_6_clone  #pytorch_1_6

#Step 3: additive noise mixing
#Inputs: Output file from Step2;  output_directory to save noisy speech; 'clean_utt_path' col_num; noise_file; SNR; 'paraphrase_pair_id' col_num;
#Outputs: Input file
#         +++ an additional column for 'noisy_utt_path'
# ---------------------------------------------------------------------------------
#  UPDATE the following variables with correct values before execution

noiseMixingFile="/nethome/achingacham/sfba4_scripts/utils/create_mixed_audio_file.py"

dir_path_suffix=$1
dir_identifiers=$2 #a space separated ids
noise_file_path=$3
SNRs=$4

clean_utt_col_num=$5  #column numbers begin with 1
pp_id_col_num=$6  #column numbers begin with 1

noise_type=`echo $noise_file_path | rev | cut -d '/' -f 1 | rev`

for dir_id in $dir_identifiers
do

  clean_utt_dir_path="$dir_path_suffix$dir_id" #use dir_path & dir_identifier
  input_file="$dir_path_suffix$dir_id/text2speech.txt" #use dir_path & dir_identifier

  noisy_utt_maindir_path="$clean_utt_dir_path""_noisy"
  mkdir $noisy_utt_maindir_path

  noisy_utt_maindir_path="$noisy_utt_maindir_path/noise_$noise_type"
  mkdir $noisy_utt_maindir_path

  for SNR in $SNRs
  do

    ### setting output args ###
    col_clean_utt="clean_utt_path-$dir_id" #colname in the output file
    col_noisy_utt="noisy_utt_path-$dir_id" #colname in the output file

    noisy_utt_outputfile_path=$noisy_utt_maindir_path/$noise_type"_"$SNR"_for_STOI.tsv"
    stoi_outputfile_path=$noisy_utt_maindir_path/$noise_type"_"$SNR"_with_STOI.tsv"
    output_col="STOI-$dir_id"
    ###

    noisy_utt_dir_path="$noisy_utt_maindir_path/snr_$SNR"
    mkdir $noisy_utt_dir_path

    tmp_id=`date +%N`

    rm  "/tmp/$tmp_id"_metadata.txt
    touch "/tmp/$tmp_id"_metadata.txt

    count=0

    while read line ;
    do

    count=$((count+1))

    clean_file_path=`echo "$line" | cut -f "$clean_utt_col_num"`

    if ! grep -q $clean_utt_dir_path <<< $clean_file_path
    then
      clean_file_path="$clean_utt_dir_path/$clean_file_path"
    fi

    if ! grep -q ".wav" <<< $clean_file_path
    then
      clean_file_path="$clean_file_path.wav"
    fi

    pp_id=`echo "$line" | cut -f "$pp_id_col_num"`
    clean_file_name=`echo "$clean_file_path" | rev | cut -d '/' -f 1 | rev`

    ### add two columns 'clean_utt_path' and 'noisy_utt_path' if not exists in the metadata
    echo -e "$noisy_utt_dir_path/$clean_file_name" >> "/tmp/$tmp_id"_metadata.txt

    if ! test -f $clean_file_path
    then
#      echo "-- $line --"
#      echo "-- $pp_id_col_num,$pp_id --"
#      echo "-- $clean_utt_col_num, $clean_file_name --"
      echo "File not exists! $clean_file_path"
      continue  #dicontinue this current iteration as clean file does not exist
    fi

    # 3 possible methods to mix noise differs in the noise snippet selection.
    # method 1: For noise mixing with a snippet that starts with a same index for all paraphrases
    python3 $noiseMixingFile -mix  --max_mix_length $pp_id --noise_file "$noise_file_path" --clean_file "$clean_file_path" --output_mixed_file "$noisy_utt_dir_path/$clean_file_name" --snr $SNR ## --output_clean_file "$input_dir_path/clean_after_NoiseMix/$utt_file_name"
    #
    # OR method 2: For noise mixing with a snippet that starts with a same index=1000
    # python3 $noiseMixingFile -mix  --max_mix_length 1000 --noise_file "$noise_file_path" --clean_file "$clean_file_path" --output_mixed_file "$noisy_utt_dir_path/$clean_file_name" --snr $SNR ## --output_clean_file "$input_dir_path/clean_after_NoiseMix/$utt_file_name"
    #
    # OR method 3: For noise mixing with a random snippet of the noise file
    # python3 $noiseMixingFile -mix --noise_file "$noise_file_path" --clean_file "$clean_file_path" --output_mixed_file "$noisy_utt_dir_path/$clean_file_name" --snr $SNR  ## --output_clean_file "$input_dir_path/clean_after_NoiseMix/$utt_file_name"

    done < <(tail -n +2 $input_file) #to skip the first line

    echo -e "$col_noisy_utt" > "/tmp/$tmp_id"_header.txt
    cat "/tmp/$tmp_id"_header.txt "/tmp/$tmp_id"_metadata.txt  > "/tmp/$tmp_id"_utt_path.txt


    paste $input_file "/tmp/$tmp_id"_utt_path.txt > $noisy_utt_outputfile_path

    echo "STOI calculation for: $noisy_utt_outputfile_path"
    stoi_script="/nethome/achingacham/sfba4_scripts/utils/get_stoi.py"

    python3 $stoi_script  -inFile "$noisy_utt_outputfile_path"  \
                      -outFile "$stoi_outputfile_path" \
                      -inFileCleanCol $col_clean_utt \
                      -inFileNoisyCol $col_noisy_utt \
                      -outSTOICol $output_col

    done

  done



rm /tmp/*_metadata.txt
rm /tmp/*_header.txt