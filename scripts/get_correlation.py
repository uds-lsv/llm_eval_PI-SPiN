import os
import csv
import ipdb
import time
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="script to find the (pearson) correlation between same column in multiple files")

    parser.add_argument('--list_input_files', '-iFs', help="a space separated list of input files which have a similar column to be compared",
                        default="/projects/SFB_A4/Corpora/TaPaCo/data/TaPaCo_testset_P1.txt-2023-08-09_02:57:27-babble_0.txt-best_hypo.txt.idx,/projects/SFB_A4/Corpora/TaPaCo/data/TaPaCo_testset_P1.txt-2023-08-09_02:57:27-babble_-3.txt-best_hypo.txt.idx"
                        )
    parser.add_argument('--input_file_delim', '-iFD', help="input file delimiter", default="\t")
    parser.add_argument('--input_file_header', '-iFH', help="input file header row", default=0, type=int)

    parser.add_argument("--input_column", "-iCol",
                        help="a column which needs to be compared across multiple files",
                        default='best_STOI-idx')

    args = parser.parse_args()
    start_time = time.time()
    exec_ts = str(time.time_ns())

    list_files = args.list_input_files.split(' ')

    for input_files in list(combinations(list_files, 2)):

        first_input_file, second_input_file = input_files

        if args.input_file_header >= 0:
            first_input_data = pd.read_table(first_input_file, sep=args.input_file_delim, quoting=csv.QUOTE_NONE,
                                       header=args.input_file_header)
            second_input_data = pd.read_table(second_input_file, sep=args.input_file_delim, quoting=csv.QUOTE_NONE,
                                             header=args.input_file_header)
        else:
            first_input_data = pd.read_table(first_input_file, sep=args.input_file_delim, quoting=csv.QUOTE_NONE,
                                       header=None)
            second_input_data = pd.read_table(second_input_file, sep=args.input_file_delim, quoting=csv.QUOTE_NONE,
                                             header=None)

        x = first_input_data[args.input_column]
        y = second_input_data[args.input_column]

        res = stats.pearsonr(x, y)
        ipdb.set_trace()
        print("Correlation btw: \n", first_input_file, "\n", second_input_file)
        print(res)


