
#a python module to generate the following sentence-level metrics:
import os
import re
import nltk
import ipdb
import time
import argparse
import numpy as np
import pandas as pd

from g2p_en import G2p
from sfba4.utils import causalLM
from sfba4.utils import plotGraphs as pG

from matplotlib import pyplot as plt
from scipy import stats

def make_2D_plot(x_values=None, y_values=None,
                 x_range=None, y_range=None,
                 x_label=None, y_label=None,
                 plot_save_path="/tmp/tmp.jpg"):

    vmax = len(x_values) * 0.1  # 10% of the whole set
    if (x_range is not None) and (y_range is not None):
        h = plt.hist2d(x_values, y_values,
                       vmax=vmax,
                       #density=True,
                       range=[x_range, y_range])
    else:
        h = plt.hist2d(x_values, y_values,
                       vmax=vmax,
                       #density=True
                       )

    plt.colorbar(h[3])
    #ipdb.set_trace()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Distribution of pairs in the dataset")
    plt.savefig(plot_save_path)
    plt.close()


def make_plots(input_data_frame, list_interesting_columns, output_dir, list_popmean=None):

    output_dict = {}

    list_interesting_columns = [col for col in list_interesting_columns if col in input_data_frame.columns]

    if list_popmean is None:
        list_popmean = [1.0] * len(list_interesting_columns)


    for each_idx, each_if in enumerate(list_interesting_columns):

        list_values = input_data_frame[each_if]
        mean_value = round(np.mean(list_values), 3)
        std_value = round(np.std(list_values), 3)
        sem_value = round(stats.sem(list_values), 3)

        one_sample_ttest = stats.ttest_1samp(list_values, popmean=list_popmean[each_idx])

        output_dict[each_if] = {'sample mean': mean_value,
                                'sample SD': std_value,
                                'sample SEM': sem_value,
                                'pop mean': list_popmean[each_idx],
                                'ttest': one_sample_ttest}

        plot_title = "Distribution of " + each_if + "(mu = "+str(mean_value)+")" + "\n ttest(popmean=" + str(list_popmean[each_idx]) + "): p-value=" + str(one_sample_ttest[1])
        # pG.plot_histogram(list_values, xlabel=each_if, ylabel="Number of pairs",
        #                   title=plot_title, fileName=output_dir + "/histogram_" + each_if)

        #ipdb.set_trace()

    # Make Density plots of paraphrases as a function of semantic similarity and lexical diversity.

    list_x_cols = ['Bertscore-f1', 'Bertscore-f1']  #, 'Bertscore-f1', 'Bertscore-f1', 'ratio_ppl', 'ED_ph', 'STOI-input_text']
    list_y_cols = ['LD', 'ED_ph']  #, 'ratio_ppl', 'best_ratio_STOI', 'best_ratio_STOI', 'best_ratio_STOI', 'best_ratio_STOI']

    feat2range = {'Bertscore-f1': [0.7, 1.0],
                  'LD': [0.0, 1.0],
                  'ED_ph': [0.0, 1.0],
                  #'ratio_ppl': [0.0, 6.0],
                  #'best_ratio_STOI': [0.5, 1.5],
                  #'ratio_STOI': [0.0, 2.0],
                  }

    for x_col, y_col in zip(list_x_cols, list_y_cols):

        # x_col = 'Bertscore-f1'
        # y_col = "LD"


        x_values = input_data_frame[x_col].to_list()
        y_values = input_data_frame[y_col].to_list()

        if x_col in feat2range:
            x_range = feat2range[x_col]
        else:
            x_range = [min(x_values), max(x_values)]

        if y_col in feat2range:
            y_range = feat2range[y_col]
        else:
            y_range = [min(y_values), max(y_values)]

        make_2D_plot(x_values, y_values,
                     x_label=x_col,
                     y_label=y_col,
                     x_range=x_range,
                     y_range=y_range,
                     plot_save_path=output_dir + "/hist2D_" + x_col + "-" + y_col  + ".jpg")

        #ipdb.set_trace()

    return output_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser("a python module for calculating metrics related to paraphrase generation task.")

    parser.add_argument("--input_file", '-iF', help="input file which lists the sentences")
    parser.add_argument("--input_file_delim", '-iFD', help="input file delimiter", default="\t")
    parser.add_argument("--input_file_header", '-iFH', default=None, type=int, help="input file's header row number; default(None)")
    parser.add_argument("--log_dir", '-lD', default="../logs", help="a log directory which stores configs and misc logs")
    parser.add_argument('--plot_interesting_cols', '-pIC', default="Bertscore-f1,max_ratio_ppl,LD,ED_bow,max_ratio_n_chars,min_ratio_n_phonemes,min_ratio_ppl", help="a comma separated list of columns (in data_pairs) to plot")
    parser.add_argument('--plot_interesting_cols_popmean', '-pIC-PM', default="0.85,1.0,0.5,1.0,1.0,1.0,1.0", help="a comma separated list of popmeans for ttest")
    parser.add_argument('--plot_grouping_cols', '-pGC', default=None, help="a comma separated list of columns (in data_pairs) to group plots")

    args = parser.parse_args()
    dict_args = vars(args)

    exec_ts_str = time.strftime("%Y%m%d-%H%M%S")
    args.log_dir = os.path.join(args.log_dir, exec_ts_str)
    os.mkdir(args.log_dir)

    with open(os.path.join(args.log_dir, "configs"), "w") as cFile:
        for key, value in dict_args.items():
            cFile.write("\n" + str(key) + "\t" + str(value))


    input_data = pd.read_table(args.input_file, sep=args.input_file_delim, header=args.input_file_header)
    list_interesting_feats = args.plot_interesting_cols.split(',')  #['Bertscore-f1', 'LD']
    list_interesting_feats_popmean = [flost(i) for i in args.plot_interesting_cols_popmean.split(',')]

    if args.plot_grouping_cols is None:

        out_dict = make_plots(input_data_frame=input_data, list_interesting_columns=list_interesting_feats,
                   output_dir=args.log_dir, list_popmean=list_interesting_feats_popmean)

        with open(os.path.join(args.log_dir, "configs"), "a") as cFile:
            for key, value in out_dict.items():
                cFile.write("\n" + str(key) + "\t" + str(value))

    else:
        # make sure grouping is possible, first.
        grouping_cols = args.plot_grouping_cols.split(',')
        assert all([True if gc in input_data.columns else False for gc in grouping_cols])

        for grp_idx, grp_df in input_data.groupby(grouping_cols):
            if not os.path.exists(args.log_dir+"_"+str(grp_idx)):
                os.mkdir(args.log_dir+"_"+str(grp_idx))

            out_dict = make_plots(input_data_frame=grp_df, list_interesting_columns=list_interesting_feats,
                       output_dir=args.log_dir+"_"+str(grp_idx), list_popmean=list_interesting_feats_popmean)

            with open(os.path.join(args.log_dir, "configs"), "a") as cFile:
                cFile.write("Group_idx: "+str(grp_idx))
                for key, value in out_dict.items():
                    cFile.write("\n" + str(key) + "\t" + str(value))


