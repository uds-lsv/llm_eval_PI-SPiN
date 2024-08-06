import os
import ipdb
import time
import argparse

import numpy as np
import pandas as pd
from sent_comp import get_features

def generate_acoustic_features(dargs, **kwargs):

    start_time = time.time()
    input_file = dargs['inputFile']
    input_file_sep = dargs['inputFileSeparator']
    input_file_clean = dargs['inputFileCleanAudioCol']
    input_file_noisy = dargs['inputFileNoisyAudioCol']
    output_file_stoi = dargs['outputFileSTOICol']

    #ipdb.set_trace()
    input_data = pd.read_table(input_file, sep=input_file_sep, quoting=3, na_filter=False)  # header=None
    input_data = input_data.dropna() #drop the items with NaN values

    #ipdb.set_trace()
    input_data[output_file_stoi] = input_data.apply(lambda row: get_features.get_STOI(row[input_file_clean], row[input_file_noisy]), axis=1)

    end_time = time.time()
    print("Total time taken: ", round(end_time-start_time), "\t for ", len(input_data), " items")

    output_data = input_data

    # avoid all incorrect ones.
    # output_data = input_data[(input_data[output_file_stoi] <= 1.0) & (input_data[output_file_stoi] >= -1.0)]
    return output_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='a small script to get STOI from already created audio files')
    parser.add_argument("-inFile", '--inputFile',
                        default="/projects/SFB_A4/Corpora/COCA_audio/text2speech_forSTOI_top250.txt",
                        help="input file with a list of short sentences/single words")
    parser.add_argument("-inFileSep", '--inputFileSeparator', help="input file delimiter", default="\t")
    parser.add_argument("-inFileText", '--inputFileTextCol', default='word',
                        help="input file's column to get text transcription")
    parser.add_argument("-inFileCleanCol", '--inputFileCleanAudioCol', default='clean_utt_path',
                        help="input file's column to get clean audio file")
    parser.add_argument("-inFileNoisyCol", '--inputFileNoisyAudioCol', default='noisy_utt_path',
                        help="input file's column to get noisy audio file")
    parser.add_argument("-outSTOICol", '--outputFileSTOICol', default='STOI',
                        help="output file's column to store STOI values")
    parser.add_argument("-outFile", '--outputFile', help="output file with features for each entry in the input file",
                        default=None)

    args = parser.parse_args()
    dict_args = vars(args)
    output_data = generate_acoustic_features(dict_args)

    if args.outputFile is None:
        print("Set args.outputFile a value!")
        args.outputFile = args.inputFile + "_with_STOI"

    #ipdb.set_trace()
    output_data.to_csv(args.outputFile, sep="\t", index=False)

    #ipdb.set_trace()
    #print("STOI calculation is over. Next phonemes")

    #generate (high-level) phonetic features:
    #feature_set = get_features.FeatureSet()
    #output_data = output_data.apply(feature_set.generate_feature_set_2, utt_col=args.inFileText, axis=1)
    #output_data.to_csv(args.outputFile, sep="\t", index=False)
    #ipdb.set_trace()
