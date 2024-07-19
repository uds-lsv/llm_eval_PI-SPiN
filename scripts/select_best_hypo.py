import os
import re
import ipdb
import csv
import time
import argparse

import numpy as np
import pandas as pd
from scipy import stats

def report_stats(input_dataframe, list_columns=None, list_columns_popmean=None, lFile=None):

    for each_col, each_col_pm in zip(list_columns, list_columns_popmean):
        col_values = input_dataframe[each_col]
        if lFile is None:
            print("Column: " + each_col)
            print("No of items: " + str(len(input_dataframe)))
            print("Mean: ", col_values.mean())
        else:
            lFile.write("\nColumn: " + each_col)
            lFile.write("\nNo of items: " + str(len(input_dataframe)))
            lFile.write("\nMean: " + str(col_values.mean()))

        if each_col_pm is not None:
            col_values_above_mean = input_dataframe[input_dataframe[each_col] > each_col_pm][each_col]
            ttest = stats.ttest_1samp(col_values, popmean=each_col_pm)
            if lFile is None:
                print("Given popmean: " + str(each_col_pm) + ", " + str(ttest))
                print("Mean of items above popmean:" + str(col_values_above_mean.mean()))
            else:
                lFile.write("\nGiven popmean: " + str(each_col_pm) + ", " + str(ttest))
                lFile.write("\nMean of items above popmean:" + str(col_values_above_mean.mean()))

        print(" ##### ")

    #return mean, mean_above_popmean, ttest

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="script to select a hypothesis based on pair-wise features (ratio/diff/similarity scores  btw hypothesis and input)")

    parser.add_argument('--input_file', '-iF', help="input file which has pairs of source and hypotheses on each line",
                        default="/projects/SFB_A4/Corpora/quora-question-pairs/dataset_PG_all_ord_phLen/checkpoints-base-e50-metricloss/test.hypo_checkpoint_best.pt.tsv_nbests.tsv_100"
                        )
    parser.add_argument('--input_file_delim', '-iFD', help="input file delimiter", default="\t")

    parser.add_argument('--input_file_header', '-iFH', help="input file header row", default=0, type=int)

    parser.add_argument("--input_column", "-iCol", help="a column which needs to be compared repeatedly with other columns at each line",
                        default='input_text')

    parser.add_argument("--input_file_filter", "-iFF",
                        help="another file with the same input_column name to filter items in the input data",
                        default='/projects/SFB_A4/A4-IntelligibleParaphrasesinNoise/data/HE_300_600/SWDA_short_utterances_300_600.tsv-08-11-2023,233716_split_cols-babble_-5.txt-best_hypo.txt_stimuli_top_30.txt')

    parser.add_argument("--comparison_columns", "-cCols",
                        help="a space separated list of comparison columns, ie: model hypotheses",
                        default="system_response_1"
                        )
    parser.add_argument("--compare_with_input", "-cWI",
                        help="whether to perform comparison with the 'input_text'",
                        action='store_true'
                        )

    parser.add_argument("--additional_columns", "-aCols",
                        help="a comma separated list of additional columns, ie: additional features",
                        default=None
                        )

    parser.add_argument("--additional_pairwise_columns", "-aPairCols",
                        help="a comma separated list of additional columns, ie: additional pairwise features",
                        default="Words,clean_utt_path,noisy_utt_path" #None
                        )

    parser.add_argument("--criteria_column", '-criCol',
                        help="the column to be used for selecting the (best) hypothesis from nbests",
                        default='STOI')
    parser.add_argument("--criteria_function", '-criFun',
                        help="the function to be executed on criteria_column values, to select best hypothesis among nbests. Values: 'max'(default)/'min'",
                        default='max')
    parser.add_argument("--criteria_column_error", '-criColErr',
                        help="the error value in the criteria column to be used for ignoring hypothesis while selection",
                        default=-9999,
                        type=float)

    parser.add_argument('--outputFile', "-oF",
                        help="output file with selected hypothesis and all pairwise values",
                        default=None)

    args = parser.parse_args()
    start_time = time.time()
    exec_ts = str(time.time_ns())

    if args.outputFile is None:
        print("Empty output file.")
        raise NotImplementedError

    if args.input_file_header >= 0:
        input_data = pd.read_table(args.input_file, sep=args.input_file_delim, quoting=csv.QUOTE_NONE,
                                   header=args.input_file_header)
    else:
        input_data = pd.read_table(args.input_file, sep=args.input_file_delim, quoting=csv.QUOTE_NONE,
                                   header=None)

    #ipdb.set_trace()
    list_interested_columns = []
    for each_col in input_data.columns:
        each_col_flag = 1
        for cc in args.comparison_columns.split(' '):
            #if it exists, then ignore
            if re.findall(cc, each_col):
                each_col_flag = 0

        if each_col_flag:
            list_interested_columns.append(each_col)

    if args.compare_with_input:
        args.comparison_columns = args.comparison_columns + " " + args.input_column

    # find relevant columns
    # compare based on the given criteria
    # 'idx' value ranges from 1 ... n (eg: system_response_1, system_response_2 ... system_response_n)
    if args.criteria_function == 'max':
        input_data[args.criteria_column + "-best_hypo"] = input_data.apply(lambda row: max([row[args.criteria_column+"-"+cc] for cc in args.comparison_columns.split(' ')]), axis=1)
        input_data[args.criteria_column + "-best_hypo-idx"] = input_data.apply(lambda row: np.argmax([row[args.criteria_column+"-"+cc] for cc in args.comparison_columns.split(' ')]), axis=1)

    if args.criteria_function == 'min':
        input_data[args.criteria_column + "-best_hypo"] = input_data.apply(lambda row: min([row[args.criteria_column+"-"+cc] for cc in args.comparison_columns.split(' ')]), axis=1)
        input_data[args.criteria_column + "-best_hypo-idx"] = input_data.apply(lambda row: np.argmin([row[args.criteria_column+"-"+cc] for cc in args.comparison_columns.split(' ')]), axis=1)

    input_data['best_hypo'] = input_data.apply(
        lambda row: [row[cc] for cc in args.comparison_columns.split(' ')]
        [row[args.criteria_column + "-best_hypo-idx"]], axis=1)

    list_interested_columns.append('best_hypo')
    list_interested_columns.append(args.criteria_column + "-best_hypo")
    list_interested_columns.append(args.criteria_column + "-best_hypo-idx")

    if args.additional_pairwise_columns is not None:
        for add_cols in args.additional_pairwise_columns.split(','):
            input_data[add_cols + "-best_hypo"] = input_data.apply(
                lambda row: [row[add_cols + "-" + cc] for cc in args.comparison_columns.split(' ')][
                    row[args.criteria_column + "-best_hypo-idx"]], axis=1)

            list_interested_columns.append(add_cols + "-best_hypo")

    # remove multiple "+ or '+
    for each_col in input_data.columns:
        if input_data[each_col].dtype == "object": #for strings
            input_data[each_col] = input_data[each_col].apply(lambda txt: re.sub("'{2,}", "", re.sub(r'"{2,}', '', txt)))

    # ipdb.set_trace()
    # report best hypothesis and related features
    input_data['best_hypo_ratio_' + args.criteria_column] = input_data.apply(
        lambda row: row[args.criteria_column + "-best_hypo"] / row[args.criteria_column + "-" + args.input_column], axis=1)
    input_data['best_hypo_diff_' + args.criteria_column] = input_data.apply(
        lambda row: row[args.criteria_column + "-best_hypo"] - row[args.criteria_column + "-" + args.input_column], axis=1)

    list_interested_columns.append('best_hypo_ratio_' + args.criteria_column)
    list_interested_columns.append('best_hypo_diff_' + args.criteria_column)

    sorted_input_data = input_data.sort_values('best_hypo_ratio_' + args.criteria_column, ascending=False)
    input_data_interested_rows = input_data[input_data['best_hypo_ratio_' + args.criteria_column] != 1.0]

    input_data.to_csv(args.outputFile, sep="\t", index=False)

    input_data[args.criteria_column + "-best_hypo-idx"].to_csv(args.outputFile + ".idx", sep="\t", index=False)
    input_data[list_interested_columns].to_csv(args.outputFile + "_stimuli_all_with_trivial.txt",
                                               sep="\t",
                                               index=False)
    input_data_interested_rows[list_interested_columns].to_csv(args.outputFile + "_stimuli_all_wo_trivial.txt",
                                                               sep="\t",
                                                               index=False)

    select_n = int(len(input_data)*(10/100)) #select '10%' of evauation set

    sorted_input_data[list_interested_columns].head(select_n).to_csv(
        args.outputFile + "_stimuli_top_" + str(select_n) + ".txt", sep="\t", index=False)
    input_data[list_interested_columns].sample(n=select_n, random_state=42).to_csv(
        args.outputFile + "_stimuli_random_" + str(select_n) + ".txt", sep="\t", index=False)

    log_file = open(args.outputFile+"_stimuli.log", "w")

    print("############## All pairs #########")
    log_file.write("\n############## All pairs #########")

    # remove all those empty hypothesis before numerical evaluation
    input_data = input_data[input_data['best_hypo'] != '-']

    report_stats(input_data,
                 list_columns=['best_hypo_ratio_' + args.criteria_column,
                               args.criteria_column + "-" + args.input_column,
                               args.criteria_column + "-best_hypo"],
                 list_columns_popmean=[1.0, None, None], lFile=log_file)

    # ipdb.set_trace()
    print("############## Selected pairs (HE) #########")
    log_file.write("\n############## Selected pairs  (HE) #########")
    filter_file = pd.read_table(args.input_file_filter)
    filter_file_items = filter_file[args.input_column]
    filtered_input_data = input_data[input_data[args.input_column].isin(filter_file_items)]

    report_stats(filtered_input_data,
                 list_columns=['best_hypo_ratio_' + args.criteria_column,
                               args.criteria_column + "-" + args.input_column,
                               args.criteria_column + "-best_hypo"],
                 list_columns_popmean=[1.0, None, None], lFile=log_file)

    filtered_input_data[list_interested_columns].to_csv(args.outputFile + "_selected_stimuli.txt",
                                                        sep="\t",
                                                        index=False)

    print("############## Top pairs #########")
    log_file.write("\n############## Top pairs #########")

    report_stats(sorted_input_data.head(select_n),
                 list_columns=['best_hypo_ratio_' + args.criteria_column,
                               args.criteria_column + "-" + args.input_column,
                               args.criteria_column + "-best_hypo"],
                 list_columns_popmean=[1.0, None, None], lFile=log_file)

    print("############## Random pairs (input != output) #########")
    log_file.write("\n############## Random pairs (input != output) #########")

    print("Total size: ", len(input_data_interested_rows))
    log_file.write("\nTotal size: " + str(len(input_data_interested_rows)))

    report_stats(input_data.sample(n=select_n, random_state=42),
                 list_columns=['best_hypo_ratio_' + args.criteria_column,
                               args.criteria_column + "-" + args.input_column,
                               args.criteria_column + "-best_hypo"],
                 list_columns_popmean=[1.0, None, None], lFile=log_file)

    print("Checkout the log file: ", log_file)