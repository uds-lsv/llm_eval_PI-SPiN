
#if a single file with all hypotheses is not present creat it first.
#paste <(cut -f 16- FILE_1.txt) <(cut -f 16- FILE_2.txt) <(cut -f 16- FILE_3.txt) <(cut -f 16- FILE_4.txt) <(cut -f 16- FILE_5.txt) > OUTPUT_FILE.txt
#input_dir="/projects/SFB_A4/Corpora/TaPaCo/data/TaPaCo_testset_P1_chatGPT_audio/gTTS-TaPaCo"  #"/projects/SFB_A4/Corpora/TaPaCo/data/"
#output_file="$input_dir/withSTOI-all.txt"

script_dir="/nethome/achingacham/PycharmProjects/chatGPT/scripts/"
input_dir="/projects/SFB_A4/Corpora/TaPaCo/data/" #"/projects/SFB_A4/A4-ParaphrasesinNoise/" #"/projects/SFB_A4/Corpora/TaPaCo/data/HumanEval/chatGPT/pSTOI"
snr=-5

date_ts=`date +"%F_%T"`
output_file="$input_dir/withSTOI-$date_ts-SNR_$snr.txt"

rm $output_file
touch $output_file

for file in `ls $input_dir/TTS-2023-07-28*/withSTOI*snr_$snr.txt`  #`ls $input_dir/*tsv`
do
paste <(cut -f 1- "$output_file") <(cut -f 1- "$file") > "$output_file-temp"
cp "$output_file-temp" "$output_file"
done

sed -i 's/^\t//g' $output_file #remove the initial TAB
rm "$output_file-temp"

#input a file with both input and nbest hypothesis for selecting the best-hypothesis based on a criteria
#python3 select_best_hypo.py -iF $output_file -oF "$output_file-best_hypo.txt" -cCols "system_response_1,system_response_2,system_response_3,system_response_4_1,system_response_4_2,system_response_4_3,system_response_4_4,system_response_4_5" -criCol 'pSTOI'
python3 $script_dir/select_best_hypo.py -iF $output_file -oF "$output_file-best_hypo.txt" -cCols "system_response_1,system_response_2,system_response_3,system_response_4,system_response_5" -criCol 'STOI'
