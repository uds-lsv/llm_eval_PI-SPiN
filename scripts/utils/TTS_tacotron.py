# Following pip packages need to be installed:
# !pip install git+https://github.com/huggingface/transformers sentencepiece datasets

import time
import nltk
import sys
import ipdb
import os
import re
import argparse
import pandas as pd
import numpy as np
from num2words import num2words

import torch
import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN

import soundfile as sf
from datasets import load_dataset

# Credits to fraction_finder: https://stackoverflow.com/questions/49440525/detect-single-string-fraction-ex-%C2%BD-and-change-it-to-longer-string
import unicodedata

def fraction_finder(s):
    for c in s:
        try:
            name = unicodedata.name(c)
        except ValueError:
            continue
        if name.startswith('VULGAR FRACTION'):
            normalized = unicodedata.normalize('NFKC', c)
            numerator, _slash, denominator = normalized.partition('⁄')
            yield c, int(numerator), int(denominator)


class TTS:

    def __init__(self, processor="", model="speechbrain/tts-tacotron2-ljspeech",
                 vocoder="speechbrain/tts-hifigan-ljspeech"):

        # Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
        self.model = Tacotron2.from_hparams(source=model, savedir="tmpdir_tts")  #run_opts={"device": "cuda"}
        self.vocoder = HIFIGAN.from_hparams(source=vocoder, savedir="tmpdir_vocoder")  #run_opts={"device": "cuda"}

    def generate_wav_file(self, input_text="Hello, my dog is cute", output_file="speech.wav"):

        # execute only if the file doesn't exist already:
        if os.path.exists(output_file):
            return 1

        # execute only if the input_text is not empty:
        if input_text != "-":
            # Running the TTS
            mel_output, mel_length, alignment = self.model.encode_text(input_text)

            # Running Vocoder (spectrogram-to-waveform)
            waveforms = self.vocoder.decode_batch(mel_output)

            # Save the waverform
            old_freq = 22050
            torchaudio.save(output_file, waveforms.squeeze(1), old_freq)
            old_freq_wav, old_freq = torchaudio.load(output_file)

            transform = torchaudio.transforms.Resample(old_freq, 16000)
            #torchaudio.save(output_file, transform(old_freq_wav), 16000)
            numpy_array_amp = transform(old_freq_wav).numpy().T
            #ipdb.set_trace()
            half_sec_silence = np.expand_dims(np.zeros(8000), 1)
            numpy_array_amp = np.concatenate((half_sec_silence, numpy_array_amp, half_sec_silence))

            sf.write(output_file, numpy_array_amp, 16000)
            # torchaudio.save(output_file, waveforms.squeeze(1).to(torch.int16), 16000)  # to convert to 16-bit format


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-iFile", '--inputFile', help="path to the input data file",
                        default=None  # /projects/SFB_A4/Corpora/TaPaCo/data/TaPaCo_P1_splits/TaPaCo_testset_P1_al
                        )
    parser.add_argument("-sep", '--separator', help="delimiter for the data file(s)",
                        default="\t")
    # parser.add_argument("-hRow", '--headerRow', help="row index for the header in  data file (0 for the first line)",
    #                     default=None)
    parser.add_argument("-tCol", '--textColumn', help="name/index of input column in data file; col num starts with 1.",
                        default="Text")
    parser.add_argument("-eCols", '--extraColumns', help="name/index of additional columns in data file; col num starts with 1.",
                        default="paraphrases_id")

    parser.add_argument("-nCol", '--nameColumn', help="name/index of output filename column in data file",
                        default=None)
    parser.add_argument("-oDir", '--outputDir', help="path to the output directory",
                        default=None  # /projects/SFB_A4/Corpora/TaPaCo/data/TaPaCo_P1_splits/TaPaCo_testset_P1_al
                        )
    parser.add_argument("-oFile", '--outputFile', help="name to the output file (will be saved in outputDir)",
                        default="text2speech.txt"
                        )

    parser.add_argument("-test", '--textString', help="an input text",
                        default=None)
    parser.add_argument("-wav", '--wavFile', help="the output wav file path",
                        default=None)

    args = parser.parse_args()

    tts = TTS()

    if args.textString is not None and args.wavFile is not None:
        inputText = args.textString
        outputFile = args.wavFile
        tts.generate_wav_file(input_text=inputText, output_file=outputFile)

    #ipdb.set_trace()

    if args.textColumn.isnumeric():
        headerRow = None
        textColumn = "text-"+args.textColumn
    else:
        headerRow = 0  # else first line is the header
        textColumn = args.textColumn

    if args.inputFile is not None:
        if headerRow is None:
            input_dataset = pd.read_table(args.inputFile, sep=args.separator, header=headerRow, quoting=3)

            input_dataset = input_dataset.fillna("-")

            output_dataset = pd.DataFrame(input_dataset[int(args.textColumn) - 1].tolist(),
                                          columns=['org-'+textColumn])  # when header is None, columns are referred using indices

            for eCol in args.extraColumns.split():
                output_dataset[eCol] = input_dataset[int(eCol)-1]

        else:
            input_dataset = pd.read_table(args.inputFile, sep=args.separator, header=int(headerRow), quoting=3)

            input_dataset = input_dataset.fillna("-")

            output_dataset = pd.DataFrame(input_dataset[args.textColumn].tolist(),
                                          columns=['org-'+textColumn])  # else, just refer with the column name.

            for eCol in args.extraColumns.split():
                output_dataset[eCol] = input_dataset[eCol]


                # convert all numbers in a string to corresponding wordings
        # output_dataset['text'] = output_dataset['text'].apply(lambda txt: " ".join(
        #     [num2words(w) if w.strip('.').isnumeric() else w for w in re.sub('-', ' ', txt).split()]))

        # 'org-'+textColumn has the original text and the column  textColumn has TTS input

        # !important ignore empty strings: dropna OR fillna
        # output_dataset = output_dataset.dropna()
        list_new_txt = []

        for txt in output_dataset['org-'+textColumn].tolist():

            if txt == "-":
                list_new_txt.append("-")
            else:
                # try:
                new_txt = ""
                # for w in nltk.word_tokenize(re.sub('-', ' ', txt)): #word tokenize splits you've -> you 've; can't -> ca n't; doesn't -> doesn ' t ...
                # introduce space before & after few symbols: [, ], -, /
                for each_sym in ['-', ',', '/', '?', '[', ']']:
                    try:
                        txt = re.sub(re.escape(each_sym), " " + each_sym + " ", txt)
                    except:
                        ipdb.set_trace()

                for w in txt.split(' '):
                    if w.strip('.-,').isnumeric():
                        w = w.strip('.-,')
                        frac_flag = False
                        for ch in w:
                            if unicodedata.name(ch).startswith('VULGAR FRACTION'):
                                frac_flag = True
                                for ch, num, denom in fraction_finder(ch):
                                    new_txt = new_txt + " " + num2words(num) + " by " + num2words(denom)

                        if not frac_flag:
                            new_txt = new_txt + " " + num2words(w)
                    else:
                        new_txt = new_txt + " " + w

                list_new_txt.append(new_txt.strip())

                # if '8-ounce' in txt:
                #     ipdb.set_trace()

        output_dataset[textColumn] = list_new_txt
        list_new_txt_uniq = output_dataset[textColumn].unique()
        txt2wav = {txt: os.path.join(args.outputDir, "utt_"+str(i)+".wav") for i, txt in enumerate(list_new_txt_uniq)}
        #ipdb.set_trace()

        if args.outputDir is None:
            args.outputDir = datetime.datetime.now().strftime("%m-%d-%Y,%H%M%S")

        # output_dataset['clean_utt_path-' + textColumn] = [os.path.join(args.outputDir, 'utt_' + str(i) + '.wav')  for i in
        #                                     range(len(output_dataset))]

        output_dataset['clean_utt_path-' + textColumn] = output_dataset[textColumn].apply(
            lambda txt: txt2wav[txt])

        output_dataset['clean_utt_path-' + textColumn] = output_dataset.apply(
            lambda row: re.sub('.wav', '_empty.wav', row['clean_utt_path-' + textColumn]) if row[textColumn] == "-" else
            row['clean_utt_path-' + textColumn], axis=1)

        output_dataset.to_csv(os.path.join(args.outputDir, args.outputFile), sep="\t", index=False)

        print("No of utterances to be generated: ", len(txt2wav))
        assert len(input_dataset) == len(output_dataset)

        # ipdb.set_trace()
        for in_txt, out_wav in txt2wav.items():
            tts.generate_wav_file(input_text=in_txt, output_file=out_wav)

        # output_dataset.apply(
        #     lambda row: tts.generate_wav_file(input_text=row[textColumn], output_file=row['clean_utt_path-'+textColumn]), axis=1)