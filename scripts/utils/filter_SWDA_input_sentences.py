import re
import ipdb
import sys
import argparse
import pandas as pd
from nltk import word_tokenize as wt

def remove_chars_within_brackets(test_str):
    ret = ''
    skip1c = 0
    skip2c = 0
    skip3c = 0
    skip4c = 0

    for i in test_str:
        if i == '[':
            skip1c += 1
        elif i == '(':
            skip2c += 1
        elif i == '{':
            skip3c += 1
        elif i == '<':
            skip4c += 1
        elif i == ']' and skip1c > 0:
            skip1c -= 1
        elif i == ')'and skip2c > 0:
            skip2c -= 1
        elif i == '}' and skip3c > 0:
            skip3c -= 1
        elif i == '>' and skip4c > 0:
            skip4c -= 1
        elif skip1c == 0 and skip2c == 0 and skip3c == 0 and skip4c == 0:
            ret += i
    return ret

#print(remove_chars_within_brackets( "hellow (sdfs) [awuhhh] are you?"))


input_file = sys.argv[1]
output_file = sys.argv[2]


input_data = pd.read_table( input_file, header=0, sep=",")
#input_data.columns = ['text']

input_data['clean_text'] = input_data['text'].apply( remove_chars_within_brackets )
input_data['clean_text'] = input_data['clean_text'].apply(lambda txt: txt.strip(' /-*#'))
input_data['clean_text'] = input_data['clean_text'].apply(lambda txt: '' if ']' in txt or '}' in txt or ')' in txt or '>' in txt else txt)

input_data['clean_text_org'] = input_data['clean_text'].apply(lambda txt: '' if ']' in txt or '}' in txt or ')' in txt or '>' in txt else txt)

#first remove 'uh' disfluency from the utterances ( for better paraphrsing)
input_data['clean_text'] = input_data['clean_text'].apply( lambda txt : ' '.join( [w for w in wt(txt) if w.lower() not in ['uh', 'huh', 'um', 'hm', 'hmm', 'mm']]).lower() )
#input_data['clean_text'] = input_data['clean_text'].apply( lambda txt : ' '.join( [w for w in txt.split() if w.lower() not in ['uh', 'huh', 'um', 'hm', 'hmm', 'mm']]).lower() )

# check isalpha() ?
input_data['clean_text'] = input_data['clean_text'].apply( lambda txt : ' '.join( [w for w in wt(txt) if not(not w.isalpha() and len(w)==1)]) )
#input_data['clean_text'] = input_data['clean_text'].apply( lambda txt : ' '.join( [w for w in txt.split() if not(not w.isalpha() and len(w)==1)]) )

# avoid the extra space before ' eg: they 've -> they've
input_data['clean_text'] = input_data['clean_text'].apply( lambda txt: re.sub(" '", "'", txt) )

#then mark all utterances with '*listen; or /*spelling: or *% or *
input_data['avoid_or_not'] = input_data.apply( lambda row : 1 if '*' in row['clean_text'] else 0, axis=1  ) # to avoid them in final list
input_data['avoid_or_not'] = input_data['clean_text'].apply(lambda txt : 1 if '=' in txt or '^' in txt else 0)

input_data['modified_or_not'] = input_data.apply(lambda row: 0 if row['text'] == row['clean_text'] else 1, axis=1  )
#input_data['n_words'] = input_data['clean_text'].apply(lambda txt: len( [w for w in wt(txt)] ))
input_data['n_words'] = input_data['clean_text'].apply(lambda txt: len( [w for w in txt.split()] ))

#required_set = input_data[(input_data['modified_or_not'] == 0) & (input_data['n_words'] >= 10) & (input_data['n_words'] <= 12)] # for PiN data
required_set = input_data[(input_data['avoid_or_not'] == 0) & (input_data['n_words'] >= 10) & (input_data['n_words'] <= 12)] # for data augmentation changed the length calculation (is.aplha altered)

#ipdb.set_trace()
pd.DataFrame(required_set[['clean_text', 'clean_text_org']]).to_csv(output_file, sep="\t", index=False)

