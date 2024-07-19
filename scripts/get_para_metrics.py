
#a python module to generate the following sentence-level metrics:
import os
import re
import csv
import nltk
import ipdb
import time
import argparse
import numpy as np
import pandas as pd

from collections import Counter
from num2words import num2words
from pyphonetics import RefinedSoundex

from Levenshtein import setratio
from Levenshtein import distance as lev_dist

from g2p_en import G2p
from sfba4.utils import causalLM

from phonemes2stoi.seq2regression_test import predict_scores
import plot_metrics


class ParaMetrics:

    def __init__(self, col_pair_id='pair_id', col_paired_text='paired-sentence', g2p_model=None, language_model=None, stoi_pred_model=None, tts_model=None, **config):

        self.input_data = None

        # initialize all dictionaries
        self.dict_input_data = {}
        self.dict_input_data_lex_feats = {}
        self.dict_input_data_pho_feats = {}
        self.dict_input_data_ling_feats = {}
        self.dict_input_data_acou_feats = {}

        self.data_pairs = {}
        self.data_pairs_col_id = col_pair_id
        self.data_pairs_col_text = col_paired_text

        self.language_model = language_model
        self.g2p_model = g2p_model
        self.stoi_prediction_model = stoi_pred_model

        for key, value in config.items():
            setattr(self, key, value)

    def data_preprocess_pairs(self, list_input_data_pairs, dict_add_features=None):

        list_data_pairs = []

        for group_idx, in_data_pairs in enumerate(list_input_data_pairs):

            if isinstance(in_data_pairs, list):
                dict_key = in_data_pairs
            else:
                dict_key = in_data_pairs.tolist()
                #dict_key = sorted(in_data_pairs) #keep the pairs as it is; don't change Source to a tagret.

            list_data_pairs.append([group_idx]+dict_key)     # id and the corresponding pair element

        self.data_pairs = pd.DataFrame(list_data_pairs,
                                       columns=[self.data_pairs_col_id] + [self.data_pairs_col_text+'-'+str(i) for i in range(2)]) #  for pairs

        list_unique_sentences = self.data_pairs[self.data_pairs_col_text+'-0'].tolist() + self.data_pairs[self.data_pairs_col_text+'-1'].tolist()

        self.data_pairs[self.data_pairs_col_text + '-clean-0'] = self.data_pairs[self.data_pairs_col_text + '-0'].apply(
            lambda t: clean_tokens(t, tokenize=False))

        self.data_pairs[self.data_pairs_col_text + '-clean-1'] = self.data_pairs[self.data_pairs_col_text + '-1'].apply(
            lambda t: clean_tokens(t, tokenize=False))

        self.data_preprocess([i for i in set(list_unique_sentences)])

        if dict_add_features is not None:
            for each_add_feat in dict_add_features:
                feat1, feat2 = each_add_feat.split(',')
                add_feats = pd.DataFrame(dict_add_features[each_add_feat], columns=[feat1, feat2]) #  for pairs
                self.data_pairs = pd.concat([self.data_pairs, add_feats], axis=1)

                # ipdb.set_trace()

                #get their uni-directional ratio (TGT/SRC) & min/max ratios
                self.data_pairs['ratio_'+each_add_feat] = self.data_pairs.apply(lambda row: row[feat2]/row[feat1], axis=1)
                self.data_pairs['min_ratio_'+each_add_feat] = self.data_pairs.apply(lambda row: min(row[feat1]/row[feat2], row[feat2]/row[feat1]), axis=1)
                self.data_pairs['max_ratio_'+each_add_feat] = self.data_pairs.apply(lambda row: max(row[feat1]/row[feat2], row[feat2]/row[feat1]), axis=1)



    def get_semantics_paired_feats(self, bertscore_model="distilbert-base-uncased", sbert_model='stsb-roberta-large'):

        import torch
        from evaluate import load
        bertscore = load("bertscore")

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.bertscore_model = bertscore_model

        # BERTScore ~ Semantic Textual Similarity
        predictions = self.data_pairs[self.data_pairs_col_text+'-0'].tolist()
        references = self.data_pairs[self.data_pairs_col_text+'-1'].tolist()

        results = bertscore.compute(predictions=predictions, references=references,
                                    model_type=self.bertscore_model, device=device)

        self.data_pairs['Bertscore-f1'] = [round(v, 4) for v in results['f1']]  # F1 score is insenstive to the ordering within a pair.
        self.data_pairs['Bertscore-p'] = [round(v, 4) for v in results['precision']]
        self.data_pairs['Bertscore-r'] = [round(v, 4) for v in results['recall']]

        # Cosine-Similarity of sentence representions ~ STS
        # code snippet taken from: https://huggingface.co/cross-encoder/stsb-roberta-large

        # from sentence_transformers import CrossEncoder
        # model = CrossEncoder('cross-encoder/stsb-roberta-large')
        # scores = model.predict([('Sentence 1', 'Sentence 2'), ('Sentence 3', 'Sentence 4')])

        #print("Calculated all semantic features for each pair.")

    def get_linguistic_paired_feats(self, language_model='gpt2'):

        #or language_model='distilgpt2'
        #first calculate the individual 'ppl'
        self.get_linguistic_feats(language_model=language_model)

        list_ppl_scores_0 = [self.dict_input_data_ling_feats[each_sent][language_model] for each_sent in
                           self.data_pairs[self.data_pairs_col_text + '-0']]

        list_ppl_scores_1 = [self.dict_input_data_ling_feats[each_sent][language_model] for each_sent in
                           self.data_pairs[self.data_pairs_col_text + '-1']]


        self.data_pairs['ppl-0'] = list_ppl_scores_0
        self.data_pairs['ppl-1'] = list_ppl_scores_1

        #ipdb.set_trace()

        # then, aggregate the scores for each sentence. This needs to be order-insensitive. eg: absolute ratio (max_ratio/min_ratio) or absolute difference (max-min)
        self.data_pairs['max_ratio_ppl'] = self.data_pairs.apply(lambda row: round(row[['ppl-0', 'ppl-1']].max()/row[['ppl-0', 'ppl-1']].min(), 3), axis=1)
        self.data_pairs['min_ratio_ppl'] = self.data_pairs.apply(lambda row: round(row[['ppl-0', 'ppl-1']].min() / row[['ppl-0', 'ppl-1']].max(), 3), axis=1)

        # uni-directional ratio - TGT/SRC :
        self.data_pairs['ratio_ppl'] = self.data_pairs.apply(lambda row: round(row['ppl-1']/row['ppl-0'], 3), axis=1)
        # self.data_pairs['sel_ratio_ppl'] = self.data_pairs.apply(lambda row: row['ratio_ppl'] if row['ratio_ppl'] < 1.0 else 1.0, axis=1)

        #print("Calculated all linguistic features for each pair.")

    def get_lexical_paired_feats(self):

        ##	Get Words Position Deviation (WPD) and Lexical Deviation by comparing two sentences.
        ## Reference: Towards Better Characterization of Paraphrases, Timothy Liu and De Wen Soh, ACL 2022

        #first calculate the individual 'lexical features'
        self.get_lexical_feats()

        # make lists for individual columns
        sample_sentence = self.data_pairs[self.data_pairs_col_text + '-0'].iloc[0]
        list_lexical_feats = [f for f in self.dict_input_data_lex_feats[sample_sentence].keys()]

        for i in range(2): #for each item in a pair
            for each_feat in list_lexical_feats:
                self.data_pairs[each_feat + '-' + str(i)] = [self.dict_input_data_lex_feats[each_sent][each_feat] for each_sent in self.data_pairs[self.data_pairs_col_text + '-' + str(i)]]

        # then, aggregate the scores for each sentence. This needs to be order-insensitive. eg: absolute ratio (max_ratio/min_ratio) or absolute difference (max-min)
        for each_feat in list_lexical_feats:

            # ignore the feature if its not a numerical
            import numbers
            if isinstance(self.dict_input_data_lex_feats[sample_sentence][each_feat], numbers.Number):

                self.data_pairs['max_ratio_' + each_feat] = self.data_pairs.apply(lambda row:
                round(row[[each_feat + '-0', each_feat + '-1']].max() / row[[each_feat + '-0', each_feat + '-1']].min(), 3), axis=1)


                self.data_pairs['min_ratio_' + each_feat] = self.data_pairs.apply(lambda row:
                round(row[[each_feat + '-0', each_feat + '-1']].min() / row[[each_feat + '-0', each_feat + '-1']].max(), 3), axis=1)


                # uni-directional ratio - TGT/SRC :
                self.data_pairs['ratio_' + each_feat] = self.data_pairs.apply(lambda row: round(row[each_feat + '-1'] / row[each_feat + '-0'], 3), axis=1)
                # self.data_pairs['sel_ratio_' + each_feat] = self.data_pairs.apply(lambda row: row['ratio_' + each_feat] if row['ratio_' + each_feat] < 1.0 else 1.0, axis=1)


        #print( "Completed pairwise ratio calculation." )

        list_LD_scores = []
        list_WPD_scores = []
        list_norm_charlevel_ED = []

        for each_row_idx, each_row in self.data_pairs.iterrows():
            each_sent_0 = each_row[self.data_pairs_col_text + '-0']
            each_sent_1 = each_row[self.data_pairs_col_text + '-1']

            each_sent_lemmas_0 = self.dict_input_data_lex_feats[each_sent_0]['lemmas'].split()
            each_sent_lemmas_1 = self.dict_input_data_lex_feats[each_sent_1]['lemmas'].split()

            common_words = set(each_sent_lemmas_0).intersection(each_sent_lemmas_1)
            all_unique_words = set(each_sent_lemmas_0).union(each_sent_lemmas_1)

            # Lexical Deviation - calculation
            LD = round(1 - (len(common_words)/len(all_unique_words)),3)
            char_ED = round(setratio(each_sent_0.split(), each_sent_1.split()), 3)  #Reference: IBM-QCPG

            list_LD_scores.append(LD)
            list_norm_charlevel_ED.append(char_ED)

            # Word Position Deviation - calculation
            n_w_s1 = Counter(each_sent_lemmas_0)
            n_w_s2 = Counter(each_sent_lemmas_1)

            # if len(each_sent_lemmas_0) - 1 == 0 or len(each_sent_lemmas_1) - 1 == 0:
            #     raise NotImplementedError
            # ipdb.set_trace()

            normalized_wp_s1 = [round(i / (len(each_sent_lemmas_0) - 1), 2) if len(each_sent_lemmas_0)-1 !=0 else 0 for i, w in enumerate(each_sent_lemmas_0)]
            normalized_wp_s2 = [round(i / (len(each_sent_lemmas_1) - 1), 2) if len(each_sent_lemmas_1)-1 !=0 else 0 for i, w in enumerate(each_sent_lemmas_1)]

            WPD_values = []

            for each_cw in common_words:

                each_cw_pos_s1 = [i for i, w in enumerate(each_sent_lemmas_0) if w == each_cw]
                each_cw_pos_s2 = [i for i, w in enumerate(each_sent_lemmas_1) if w == each_cw]

                n_cw_s1 = n_w_s1[each_cw]
                n_cw_s2 = n_w_s2[each_cw]

                RPS_cw_s1 = sum([min([round(abs(normalized_wp_s1[s1_idx] - normalized_wp_s2[s2_idx]) / n_cw_s1, 2) for s2_idx in each_cw_pos_s2]) for s1_idx in each_cw_pos_s1])
                RPS_cw_s2 = sum([min([round(abs(normalized_wp_s2[s2_idx] - normalized_wp_s1[s1_idx]) / n_cw_s2, 2) for s1_idx in each_cw_pos_s1]) for s2_idx in each_cw_pos_s2])

                WPD_values.append(max(RPS_cw_s1, RPS_cw_s2))

            list_WPD_scores.append(round(np.mean(WPD_values), 2))

        self.data_pairs['LD'] = list_LD_scores
        self.data_pairs['ED_bow'] = list_norm_charlevel_ED
        self.data_pairs['WPD'] = list_WPD_scores

        #print("Calculated all lexical features for each pair.")

    def get_phonemic_paired_feats(self, g2p_model=None):

        #first calculate the individual 'n_phonemes'
        self.get_phonemic_feats(g2p_model=g2p_model)

        # make lists for individual columns
        sample_sentence = self.data_pairs[self.data_pairs_col_text + '-0'].iloc[0]
        list_phonemic_feats = [f for f in self.dict_input_data_pho_feats[sample_sentence].keys()]

        for i in range(2):  # for each item in a pair
            for each_feat in list_phonemic_feats:
                self.data_pairs[each_feat + '-' + str(i)] = [self.dict_input_data_pho_feats[each_sent][each_feat] for each_sent in
                                                             self.data_pairs[self.data_pairs_col_text + '-' + str(i)]]

        # then, aggregate the scores for each sentence. This needs to be order-insensitive. eg: absolute ratio (max_ratio/min_ratio) or absolute difference (max-min)
        for each_feat in list_phonemic_feats:

            # ignore the feature if its not a numerical
            import numbers
            if isinstance(self.dict_input_data_pho_feats[sample_sentence][each_feat], numbers.Number):

                self.data_pairs['max_ratio_' + each_feat] = self.data_pairs.apply(lambda row:
                        round(row[[each_feat + '-0', each_feat + '-1']].max() / row[[each_feat + '-0', each_feat + '-1']].min(), 3), axis=1)

                self.data_pairs['min_ratio_' + each_feat] = self.data_pairs.apply(lambda row:
                        round(row[[each_feat + '-0', each_feat + '-1']].min() / row[[each_feat + '-0', each_feat + '-1']].max(), 3), axis=1)


                # uni-directional ratio - TGT/SRC :
                self.data_pairs['ratio_' + each_feat] = self.data_pairs.apply(
                    lambda row: round(row[each_feat + '-1'] / row[each_feat + '-0'], 3), axis=1)
                # self.data_pairs['sel_ratio_' + each_feat] = self.data_pairs.apply(
                #     lambda row: row['ratio_' + each_feat] if row['ratio_' + each_feat] < 1.0 else 1.0, axis=1)


        list_ED_ph = []
        list_ED_ph_subs = []
        list_ED_soundex = []

        rs = RefinedSoundex()

        #calculate the Lev_Dis and divide by the max length
        for each_row_idx, each_row in self.data_pairs.iterrows():

            # print("->", each_row[self.data_pairs_col_text + '-0-clean'])
            # clean_sent_0 = ' '.join([num2words(w) if w.isnumeric() else w for w in each_row[self.data_pairs_col_text + '-0-clean'].split()])
            #
            # print("->", each_row[self.data_pairs_col_text + '-1-clean'])
            # clean_sent_1 = ' '.join([num2words(w) if w.isnumeric() else w for w in each_row[self.data_pairs_col_text + '-1-clean'].split()])
            # list_ED_soundex.append(rs.distance(clean_sent_0, clean_sent_1))

            each_sent_phonemes_0 = each_row['phonemes_wo_stress_ws-0']
            each_sent_phonemes_1 = each_row['phonemes_wo_stress_ws-1']

            max_length_pair = max(each_row['n_phonemes-0'], each_row['n_phonemes-1'])

            ph_lev_dist = lev_dist(re.sub('- ', '', each_sent_phonemes_0).split(' '),
                                   re.sub('- ', '', each_sent_phonemes_1).split(' '), weights=(1, 1, 1))

            ph_sub_dist = lev_dist(re.sub('- ', '', each_sent_phonemes_0).split(' '),
                                   re.sub('- ', '', each_sent_phonemes_1).split(' '), weights=(1, 1, 2)) #additional cost for substitution


            list_ED_ph.append(ph_lev_dist/max_length_pair)
            list_ED_ph_subs.append(ph_sub_dist/max_length_pair)


        self.data_pairs['ED_ph'] = list_ED_ph
        self.data_pairs['ED_ph_sub'] = list_ED_ph_subs
        #self.data_pairs['ED_soundex'] = list_ED_soundex

        #print("Calculated all phonemic features for each pair.")

    def get_acoustic_paired_feats(self, stoi_prediction_model=None):

        #first calculate the individual 'STOI' predictions with a given model.
        self.get_acoustic_feats(stoi_prediction_model=stoi_prediction_model)

        # make lists for individual columns
        sample_sentence = self.data_pairs[self.data_pairs_col_text + '-0'].iloc[0]
        list_acoustic_feats = [f for f in self.dict_input_data_acou_feats[sample_sentence].keys()]

        for i in range(2):  # for each item in a pair
            for each_feat in list_acoustic_feats:
                self.data_pairs[each_feat + '-' + str(i)] = [self.dict_input_data_acou_feats[each_sent][each_feat] for each_sent in
                                                             self.data_pairs[self.data_pairs_col_text + '-' + str(i)]]


        # then, aggregate the scores for each sentence. This needs to be order-insensitive. eg: absolute ratio (max_ratio/min_ratio) or absolute difference (max-min)
        for each_feat in list_acoustic_feats:

            # ignore the feature if its not a numerical
            import numbers
            if isinstance(self.dict_input_data_acou_feats[sample_sentence][each_feat], numbers.Number):

                self.data_pairs['max_ratio_' + each_feat] = self.data_pairs.apply(lambda row:
                        round(row[[each_feat + '-0', each_feat + '-1']].max() / row[[each_feat + '-0', each_feat + '-1']].min(), 3), axis=1)

                self.data_pairs['min_ratio_' + each_feat] = self.data_pairs.apply(lambda row:
                        round(row[[each_feat + '-0', each_feat + '-1']].min() / row[[each_feat + '-0', each_feat + '-1']].max(), 3), axis=1)

                # uni-directional ratio - TGT/SRC :
                self.data_pairs['ratio_' + each_feat] = self.data_pairs.apply(
                    lambda row: round(row[each_feat + '-1'] / row[each_feat + '-0'], 3), axis=1)
                # the expected TGT STOI is >= SRC STOI. That is why the expected ratio > 1.0
                # self.data_pairs['sel_ratio_' + each_feat] = self.data_pairs.apply(
                #     lambda row: row['ratio_' + each_feat] if row['ratio_' + each_feat] > 1.0 else 1.0, axis=1)

        #print("Calculated all acoustic features for each pair.")

    def data_preprocess(self, list_input_data):

        for in_data in list_input_data:
            # make sure the dictionary is not over-written
            if in_data not in self.dict_input_data:
                self.dict_input_data[in_data] = clean_tokens(in_data)

    def get_lexical_feats(self):

        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()

        self.dict_input_data_lex_feats = {}


        for in_data, in_data_tokens in self.dict_input_data.items():
            if in_data not in self.dict_input_data_lex_feats:
                self.dict_input_data_lex_feats[in_data] = {}
                self.dict_input_data_lex_feats[in_data]['sentence'] = ' '.join(in_data_tokens)

                self.dict_input_data_lex_feats[in_data]['lemmas'] = ' '.join([lemmatizer.lemmatize(w.lower()) for w in in_data_tokens])
                self.dict_input_data_lex_feats[in_data]['n_tokens'] = len(in_data_tokens)
                self.dict_input_data_lex_feats[in_data]['n_chars'] = sum([len(w) for w in in_data_tokens])
                self.dict_input_data_lex_feats[in_data]['n_chars_lw'] = np.max([len(w) for w in in_data_tokens])
                self.dict_input_data_lex_feats[in_data]['pos_lw'] = np.argmax([len(w) for w in in_data_tokens]) + 1 # positions are counted from 1.
                self.dict_input_data_lex_feats[in_data]['longest_word'] = in_data_tokens[self.dict_input_data_lex_feats[in_data]['pos_lw'] - 1]

        self.data_lexical_feats = pd.DataFrame.from_dict(self.dict_input_data_lex_feats, orient='index').reset_index()

        #print("Calculated all lexical features for each sentence.")

    def get_phonemic_feats(self, g2p_model=None):

        if g2p_model is not None:
            self.g2p_model = g2p_model

        self.dict_input_data_pho_feats = {}

        for in_data, in_data_tokens in self.dict_input_data.items():
            if in_data not in self.dict_input_data_pho_feats:
                self.dict_input_data_pho_feats[in_data] = {}
                self.dict_input_data_pho_feats[in_data]['sentence'] = ' '.join(in_data_tokens)

                #ipdb.set_trace()
                in_data_phonemes = self.g2p_model(self.dict_input_data_pho_feats[in_data]['sentence']) #!input of g2p_model is the tokenized sentence and not the original sentence.
                self.dict_input_data_pho_feats[in_data]['phonemes_ws'] = " ".join(["-" if ph == ' ' else ph for ph in in_data_phonemes])

                self.dict_input_data_pho_feats[in_data]['phonemes_wo_stress_ws'] = re.sub(r'\s\s+', ' ',
                                                                                   re.sub(r'\d', '',
                                                                                   self.dict_input_data_pho_feats[in_data]['phonemes_ws']))

                self.dict_input_data_pho_feats[in_data]['n_phonemes'] = len([ph for ph in in_data_phonemes if ph != ' '])
                #ipdb.set_trace()

        self.data_phonemic_feats = pd.DataFrame.from_dict(self.dict_input_data_pho_feats, orient='index').reset_index()

        #print("Calculated all phonemic features for each sentence.")

    def get_linguistic_feats(self, language_model='gpt2'):

        #language_model='distilgpt2'
        if language_model is not None:
            self.language_model = language_model

        clm = causalLM.cLM(model_tag=self.language_model)
        self.dict_input_data_ling_feats = {}

        for in_data, in_data_tokens in self.dict_input_data.items():
            if in_data not in self.dict_input_data_ling_feats:
                self.dict_input_data_ling_feats[in_data] = {}
                self.dict_input_data_ling_feats[in_data]['sentence'] = ' '.join(in_data_tokens)

                # perplexity based on the original sentence vs tokenized and uncased sentence.
                self.dict_input_data_ling_feats[in_data][language_model] = clm.get_lm_score(in_data) #clm.get_lm_score(' '.join(in_data_tokens))

        self.data_linguistic_feats = pd.DataFrame.from_dict(self.dict_input_data_ling_feats, orient='index').reset_index()

        #print("Calculated all linguistic features for each sentence.")

    def get_acoustic_feats(self, stoi_prediction_model=None):

        if self.stoi_prediction_model is None:
            if stoi_prediction_model is None:
                return None
            else:
                self.stoi_prediction_model = stoi_prediction_model

        if self.dict_input_data_pho_feats == {}:
            self.get_phonemic_feats()


        list_phoneme_sequences = [in_data_pho_feats['phonemes_wo_stress_ws'] for in_data, in_data_pho_feats in self.dict_input_data_pho_feats.items()]
        list_STOI_predictions = predict_scores(list_inputs=list_phoneme_sequences, modelPath=self.stoi_prediction_model)


        for in_data, pred_stoi in zip(self.dict_input_data_pho_feats.keys(), list_STOI_predictions):
            self.dict_input_data_acou_feats[in_data] = {}
            self.dict_input_data_acou_feats[in_data]['pSTOI'] = round(pred_stoi, 3)  #STOI prediction for babble SNR -5


def clean_tokens(input_string, remove_quotes=True, lower_case=True, tokenize=True):


    # strip out all quotes; only at extreme ends
    if remove_quotes:

        for _ in range(2):
            input_string = input_string.strip('"')
            input_string = input_string.strip("'")

        #re.sub("'", "", re.sub('"', '', input_string))
        # while input_string[0] == '"' or input_string[0] == "'" or input_string[-1] == '"' or input_string[-1] == "'":
        #     input_string = input_string.strip("'")
        #     input_string = input_string.strip('"')

    if lower_case:
        input_string = input_string.lower()

    if tokenize:
        return nltk.word_tokenize(input_string)
    else:
        return input_string

def get_gt_ratio_diff(row, for_col='STOI', based_on_col='PhER', fun='ratio'):
    # ipdb.set_trace()
    if for_col+'-0' not in row.index or based_on_col+'-0' not in row.index or for_col+'-1' not in row.index or based_on_col+'-1' not in row.index:
        return None
    else:

        # step 1: get gold truth scores for paired item-0 and item-1
        # step 2: then calculate gt_ratio as,: a/b if item-0 > item-1 else b/a.
        if fun == 'ratio':
            if row[based_on_col+'-0'] > row[based_on_col+'-1']:
                return row[for_col + '-0'] / row[for_col + '-1']
            else:
                return row[for_col + '-1'] / row[for_col + '-0']

        if fun == 'diff':
            if row[based_on_col + '-0'] > row[based_on_col + '-1']:
                return row[for_col + '-0'] - row[for_col + '-1']
            else:
                return row[for_col + '-1'] - row[for_col + '-0']

        # step 3: repeat this for all gt_required columns


if __name__ == '__main__':

    parser = argparse.ArgumentParser("a python module for calculating metrics related to paraphrase generation task.")

    parser.add_argument("--test_sentence", '-tS', help="an input sentence for testing",
                        default='If you have a nose that gets stuffy at night, what do you do?')
    parser.add_argument("--test_sentence_pair", '-tSP', help="an input sentence pair for testing (tab delimited)",
                        default="Your fever has only just come down so don't overexert yourself.\tYou've only just gotten over your fever so don't overdo it.")

    parser.add_argument("--input_file", '-iF', help="input file which lists the sentences")
    parser.add_argument("--input_file_delim", '-iFD', help="input file delimiter", default="\t")
    parser.add_argument("--input_file_header", '-iFH', default=None, type=int, help="input file's header row number; default(None)")
    parser.add_argument("--input_file_sent_col", '-iF-sC', default=0, help="column in the input file which lists sentences (if integer, start counting from 0)")
    parser.add_argument("--input_file_paired_sent_col", '-iF-psC', default=None, help="a comma separated list of column(s) in the input file, which lists (multiple) paired sentences (if integer, start counting from 0)")
    parser.add_argument("--input_file_pair_id_col", '-iF-pidC', default=None, help="a column in the input file which lists pair ID")
    parser.add_argument("--input_file_groundtruth_col", '-iF-gtC', default='PhER',
                        help="column in the input file which lists the metric to identify best sentence in a pair")
    parser.add_argument("--input_file_verify_cols", '-iF-vCs', default=None, help="a comma separated list of columns in the input file which lists scores. eg.STOI")
    parser.add_argument("--input_file_include_cols", '-iF-iCs', default=None,
                        help="a comma separated list of columns in the input file which needs to be included in the final output")
    parser.add_argument("--add_features", "-aF", default=None, help="a list of paired columns to add existing features. eg: src-0,STOI-p1;hypo-0,STOI-p2  ")
    parser.add_argument("--log_dir", '-lD', default="/nethome/achingacham/PycharmProjects/para_metrics/logs", help="a log directory which stores configs and misc logs")
    parser.add_argument('--plot_interesting_cols', '-pIC', default="Bertscore-f1,LD,ED_ph,n_phonemes-0,n_phonemes-1,ratio_n_phonemes,ppl-0,ppl-1,ratio_ppl,ratio_STOI,ratio_pSTOI", help="a comma separated list of columns (in data_pairs) to plot")
    parser.add_argument('--plot_grouping_cols', '-pGC', default=None,
                        help="a comma separated list of columns (in data_pairs) to group plots")

    args = parser.parse_args()
    dict_args = vars(args)

    exec_ts = time.time_ns()
    exec_ts_str = time.strftime("%Y%m%d-%H%M%S")

    # Step 0: log the configs
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    args.log_dir = os.path.join(args.log_dir, exec_ts_str)
    os.mkdir(args.log_dir)

    print("Checkout the log directory created: ", args.log_dir)

    # Step 1: make an instance of ParaMetrics (configuring a set of trained models. eg: G2P, LM, STOI-pred, TTS etc..)
    pm = ParaMetrics()
    list_input_pairs = []
    list_input_sentences = []
    dict_add_features = {} #for additional fetaures

    if args.add_features is not None:
        for each_feat_pair in args.add_features.split(';'):
            dict_add_features[each_feat_pair] = []

    # if the header row is None, all column headers are integers
    if args.input_file_header is None:
        args.input_file_sent_col = int(args.input_file_sent_col)

    input_data = pd.read_table(args.input_file, sep=args.input_file_delim, header=args.input_file_header, quoting=csv.QUOTE_NONE) #treat quote chars like ', " as it is.
    #ipdb.set_trace()

    if args.input_file_header is None:
        if "," in args.input_file_paired_sent_col:
            input_file_paired_sent_columns = [int(c) for c in args.input_file_paired_sent_col.split(",")]
            #args.input_file_paired_sent_col = int(args.input_file_paired_sent_col)

    if args.plot_grouping_cols is not None:
        grouping_cols = args.plot_grouping_cols.split(',')
        assert all([True if gc in input_data.columns else False for gc in grouping_cols])
        list_input_data = [(str(grp_idx), grp_df) for grp_idx, grp_df in input_data.groupby(grouping_cols)]

    else:
        list_input_data = [("", input_data)]

    # make plots and log dirs for each group separately
    main_log_dir = args.log_dir
    for grp_idx, input_data in list_input_data:
        #print("Group idx: ", grp_idx)
        args.log_dir = os.path.join(main_log_dir, grp_idx)
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)

        # For verification, find the aggregated scores of existing columns
        if args.input_file_verify_cols is not None:
            list_verify_cols = [each_vc for each_vc in args.input_file_verify_cols.split(',') if each_vc in input_data.columns]
        else:
            list_verify_cols = []

        # if pairs are available, make a list of them
        if args.input_file_sent_col is not None and args.input_file_pair_id_col is not None:
            list_input_pairs = [gdf[args.input_file_sent_col].tolist() for gid, gdf in input_data.groupby(args.input_file_pair_id_col)]

            # remove all pairs if either of them is empty
            list_input_pairs = [[str_a, str_b] for str_a, str_b in list_input_pairs if
                                str_a.strip() != '' and str_b.strip() != '']


            for feat_pair in dict_add_features:
                feat1, feat2 = each_feat_pair.split(',')  #if pair-id is provided instead of paired sentence, use -aF "STOI,"

                list_add_feat_values =  [gdf[feat1].tolist() for gid, gdf in input_data.groupby(args.input_file_pair_id_col) if all([i!='' for i  in gdf[args.input_file_sent_col].tolist()])]
                dict_add_features[feat_pair] = list_add_feat_values

            for each_vc in list_verify_cols:
                #print("Aggregated score for the column: ", each_vc)

                max_ratio = [gdf[each_vc].max() / gdf[each_vc].min() for _, gdf in
                             input_data.groupby(args.input_file_pair_id_col)]

                min_ratio = [gdf[each_vc].min() / gdf[each_vc].max() for gid, gdf in
                             input_data.groupby(args.input_file_pair_id_col)]

                #print(" Maximum ratio: ", round(np.mean(max_ratio), 3))
                #print(" Minimum ratio: ", round(np.mean(min_ratio), 3))

            #ipdb.set_trace()

        elif args.input_file_sent_col is not None and args.input_file_paired_sent_col is not None:
            list_input_pairs = [row[[args.input_file_sent_col, args.input_file_paired_sent_col]] for row_id, row in
                                input_data.iterrows()]

            #remove all pairs if either of them is empty
            list_input_pairs = [[str_a, str_b] for str_a, str_b in list_input_pairs if str_a.strip() != '' and str_b.strip()!= '']

            for feat_pair in dict_add_features:
                feat1, feat2 = each_feat_pair.split(',')  #if pair-id is provided instead of paired sentence, use -aF "STOI,"

                list_add_feat_values =  [row[[feat1, feat2]] for row_id, row in input_data.iterrows() if all([i!='' for i in row[[args.input_file_sent_col, args.input_file_paired_sent_col]]])]

                dict_add_features[feat_pair] = [[f1, f2] for f1,f2 in list_add_feat_values]

        elif args.input_file_sent_col is not None:
            list_input_sentences = input_data[args.input_file_sent_col].tolist()

        else:
            print("Some missing arguments!")
            raise NotImplementedError


        # Steps set (a) : paired-sentence-level features
        # Step a.2: read paired-data
        # TEST: pm.data_preprocess_pairs([args.test_sentence_pair.split('\t')])
        pm.data_preprocess_pairs(list_input_pairs, dict_add_features)

        # Step a.3: get Semantics features
        pm.get_semantics_paired_feats()

        # Step a.4: get Linguistics features
        pm.get_linguistic_paired_feats()

        # Step a.5: get Lexical features
        pm.get_lexical_paired_feats()

        # Step a.6: get Phonetic features
        g2p = G2p()
        pm.get_phonemic_paired_feats(g2p_model=g2p)

        # Step a.7: get Acoustic features
        stoi_prediction_model = "/nethome/achingacham/PycharmProjects/phonemes_to_word_recognition/models_QQP_test/best_checkpoint_STOI_1678481571226929978.tar"
        pm.get_acoustic_paired_feats(stoi_prediction_model=stoi_prediction_model)

        # Step a.7.1: get diff/ratio for some known values:
        list_extra_cols = ['PhER', 'STOI']
        dict_extra_cols = {}
        for col in list_extra_cols:
            if col in input_data.columns:
                dict_extra_cols[col] = {}
                for si, sv in zip(input_data[args.input_file_sent_col].tolist(), input_data[col].tolist()):
                    dict_extra_cols[col][si] = sv


                pm.data_pairs[col + "-0"] = pm.data_pairs.apply(lambda row: dict_extra_cols[col][row[pm.data_pairs_col_text+'-0']],
                                                                axis=1)
                pm.data_pairs[col + "-1"] = pm.data_pairs.apply(lambda row: dict_extra_cols[col][row[pm.data_pairs_col_text+'-1']],
                                                                axis=1)

                pm.data_pairs['abs_diff_' + col] = pm.data_pairs.apply(lambda row: abs(row[col+'-0'] - row[col+'-1']),
                                                                axis=1)

                pm.data_pairs['max_ratio_' + col] = pm.data_pairs.apply(
                    lambda row: max(row[col + '-0'] / row[col + '-1'], row[col + '-1'] / row[col + '-0'] ), axis=1)

                pm.data_pairs['min_ratio_' + col] = pm.data_pairs.apply(
                    lambda row: min(row[col + '-0'] / row[col + '-1'], row[col + '-1'] / row[col + '-0']), axis=1)

                args.plot_interesting_cols = args.plot_interesting_cols + ',abs_diff_' + col + ',max_ratio_' + col + ',min_ratio_' + col

        # Step a.7.2: get goldtruth_ratio(gt_ratio) based on known value (eg:PhER):
        goldtruth_col = args.input_file_groundtruth_col
        list_gt_ratios_required = ['pSTOI', 'STOI', 'n_phonemes', 'ppl']


        if goldtruth_col+'-0' in pm.data_pairs.columns and goldtruth_col+'-1' in pm.data_pairs.columns :
            for each_col in list_gt_ratios_required:
                if each_col+'-0' in pm.data_pairs.columns and each_col+'-1' in pm.data_pairs.columns:
                    #ipdb.set_trace()
                    pm.data_pairs['gt_ratio_' + each_col] = pm.data_pairs.apply(
                        lambda row: get_gt_ratio_diff(row, each_col, goldtruth_col, 'ratio'), axis=1)

                    pm.data_pairs['gt_diff_' + each_col] = pm.data_pairs.apply(
                        lambda row: get_gt_ratio_diff(row, each_col, goldtruth_col, 'diff'), axis=1)

                    args.plot_interesting_cols = args.plot_interesting_cols + ',gt_ratio_' + each_col + ',gt_diff_' + each_col

        #ipdb.set_trace()
        # Step a.8 : logging and plotting
        ipdb.set_trace()
        if args.input_file_include_cols is not None:
            for each_include_col in args.input_file_include_cols.split(','):
                pm.data_pairs[each_include_col] = input_data[each_include_col]

        pm.data_pairs.to_csv(os.path.join(args.log_dir, "output_pairs_feats.tsv"), sep="\t", index=False)

        out_dict = {}
        if len(list_input_pairs) > 0:
            list_interesting_feats = args.plot_interesting_cols.split(',')  # eg: ['Bertscore-f1','max_ratio_ppl','LD']
            out_dict = plot_metrics.make_plots(input_data_frame=pm.data_pairs, list_interesting_columns=list_interesting_feats, output_dir=args.log_dir)

        ####################################################
        # Steps set (b): sentence-level features, only if there weren't calculated earlier.
        # Step b.2: read data
        # TEST: pm.data_preprocess([args.test_sentence])

        if len(pm.data_pairs) == 0:

            pm.data_preprocess(list_input_sentences)

            # Step b.3: get Linguistic features
            pm.get_linguistic_feats(language_model='gpt2')

            # Step b.4: get Lexical features
            pm.get_lexical_feats()

            # Step b.5: get Phonemic features
            g2p = G2p()
            pm.get_phonemic_feats(g2p_model=g2p)

            # Step b.6: get acoustic features [TODO]

        # Step b.7: logging and plotting:
        pm.data_lexical_feats.to_csv(os.path.join(args.log_dir, "output_lex_feats.tsv"), sep="\t", index=False)
        pm.data_phonemic_feats.to_csv(os.path.join(args.log_dir, "output_pho_feats.tsv"), sep="\t", index=False)
        pm.data_linguistic_feats.to_csv(os.path.join(args.log_dir, "output_ling_feats.tsv"), sep="\t", index=False)


        # Save configs
        with open(os.path.join(args.log_dir, "configs"), "w") as cFile:
            for key, value in dict_args.items():
                cFile.write("\n" + str(key) + "\t" + str(value))

            cFile.write("\nParaMetrics - Attributes:")

            for key in ['language_model', 'g2p_model', 'bertscore_model', 'stoi_prediction_model']:
                value = pm.__dict__[key]
                cFile.write("\n" + str(key) + "\t" + str(value))

            cFile.write("\nPlotting outcomes:")
            for key, value in out_dict.items():
                cFile.write("\n" + str(key) + "\t" + str(value))

            for each_include_col in args.input_file_include_cols.split(','):
                cFile.write("\n" + each_include_col + "\t" + str(pm.data_pairs[each_include_col].mean()))


            #plot_columns = [c for c in list_interesting_feats if c in pm.data_pairs.columns]
            #cFile.write(str(pm.data_pairs[plot_columns].corr()))

        exec_end = time.time_ns()

        print("Total time taken (ns): ", exec_end-exec_ts, " for "+ str(len(pm.data_lexical_feats)) +" pairs :")

    print("Checkout the log directory created: ", args.log_dir)
    #ipdb.set_trace()
