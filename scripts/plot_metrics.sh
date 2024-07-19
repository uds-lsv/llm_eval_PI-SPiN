
root_dir="/nethome/achingacham/PycharmProjects/para_metrics/para_metrics/"

#For additional plots:
timestamp=$1  #"20230328-000200"
log_directory="/projects/SFB_A4/para-metrics/logs/$timestamp/"
additional_feats="Bertscore-f1,LD,min_ratio_n_phonemes,min_ratio_ppl,max_ratio_STOI"

output_file="/projects/SFB_A4/para-metrics/logs/$timestamp/output_pairs_feats.tsv"
python3 "$root_dir/plot_metrics.py" -iF $output_file -iFH 0 -lD $log_directory -pIC $additional_feats

bash filter_pairs.sh $timestamp
output_file="/projects/SFB_A4/para-metrics/logs/$timestamp/filtered_pairs.tsv"
python3 "$root_dir/plot_metrics.py" -iF $output_file -iFH 0 -lD $log_directory -pIC $additional_feats
