import os
import re
import csv
import ipdb
import copy
import json
import datetime

import argparse
import pandas as pd

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_API(uprompt, model_name, logFile):


    chat_response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user", "content": uprompt}
        ]
    )


    response = chat_response["choices"][0]['message']['content']

    #log the chat_response for further studies
    # Python program to write JSON
    # to a file
    # import json

    lFile = open(logFile, "a")
    json.dump(chat_response.to_dict(), lFile)

    lFile.write("\n")
    lFile.close()

    #ipdb.set_trace()
    return response

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-iFile", '--inputFile', help="path to the input data file",
                        default=None#/projects/SFB_A4/Corpora/TaPaCo/data/TaPaCo_P1_splits/TaPaCo_testset_P1_al
                        )
    parser.add_argument("-sep", '--separator', help="delimiter for the data file(s)",
                        default="\t")
    parser.add_argument("-hRow", '--headerRow', help="row index for the header in  data file (0 for the first line)",
                        type=int, default=None)
    parser.add_argument("-iCol", '--inputColumn', help="name/index of input column in data file",
                        default="text")
    parser.add_argument("-uPrompt", '--userPrompt', help="mention the user prompt required for the API call",
                        default=None#"Paraphrase the following sentence: "
                        )
    parser.add_argument("-mName", '--modelName', help="name the model for paraphrase generation",
                        default="gpt-3.5-turbo")

    parser.add_argument("-oTCol", '--outputTextColumn', help="output column for text in the output file", default="input_text")
    parser.add_argument("-oPCol", '--outputParaColumn', help="output column for paraphrases in the output file", default="system_response")
    parser.add_argument("-oPidCol", '--outputParaIdColumn', help="output column for paraphrases id in the output file", default="paraphrases_id")

    parser.add_argument("-oFile", '--outputFile', help="path to the output file", default=None)

    args = parser.parse_args()

    # set some variables before script execution:
    exec_ts = datetime.datetime.now()

    if args.outputFile is None:
        outputFile = args.inputFile + "-" + exec_ts.strftime("%m-%d-%Y,%H%M%S")
    else:
        outputFile = args.outputFile

    logFile = outputFile + ".json"

    if args.inputFile is not None:
        input_dataset = pd.read_table(args.inputFile, sep=args.separator, header=args.headerRow) #quoting=csv.QUOTE_NONE
    else:
        print("Check the path to the input file.")
        raise NotImplementedError

    if args.userPrompt is None:
        print("Check the user prompt. its not given!")
        raise NotImplementedError

    outputTextColumn = args.outputTextColumn
    outputParaColumn = args.outputParaColumn
    outputParaIdColumn = args.outputParaIdColumn

    if args.headerRow is None:
        output_dataset = pd.DataFrame(input_dataset[int(args.inputColumn)-1].tolist(), columns=[outputTextColumn]) #when header is None, columns are referred using indices
    else:
        output_dataset = pd.DataFrame(input_dataset[args.inputColumn].tolist(), columns=[outputTextColumn]) # else, just refer with the column name.

    if outputParaIdColumn not in input_dataset.columns:
        output_dataset[outputParaIdColumn] = output_dataset.index
    else:
        output_dataset[outputParaIdColumn] = input_dataset[outputParaIdColumn]

    #ipdb.set_trace()

    #remove additional spaces in textColumn
    output_dataset[outputTextColumn] = output_dataset[outputTextColumn].apply(lambda txt: re.sub(r' {2,}',' ', txt))
    #ipdb.set_trace()

    # generate the user prompt content:
    output_dataset['user_prompt'] = output_dataset[outputTextColumn].apply(lambda txt: args.userPrompt + txt )

    output_dataset[outputParaColumn] = output_dataset['user_prompt'].apply(lambda uprompt: call_API(uprompt, args.modelName, logFile))

    exec_te = datetime.datetime.now()

    print("Total time taken (ms): ", exec_te-exec_ts)
    output_dataset['user_prompt'] = output_dataset['user_prompt'].apply(lambda txt: re.sub(r'\n', r'\\n', txt))
    output_dataset.to_csv(outputFile, sep="\t", index=False)

    output_dataset_copy = output_dataset.copy(deep=True)

    # when multiple responses are requested, a list in returned with '\n' delimiter is the output
    new_output_dataset_diffrow = []
    new_output_dataset_diffcol = []

    n_sys_responses = []

    if '\n' in output_dataset[outputParaColumn].iloc[0]:
        for row_idx, each_row in output_dataset.iterrows():
            input_text = each_row[outputTextColumn]
            user_prompt = each_row['user_prompt']
            systems_responses = [each_resp.split(' ', 1)[1] for each_resp in re.sub('\n+', '\n', each_row[outputParaColumn]).split('\n') if len(each_resp.split(' ', 1))==2] #the first split is the index

            if row_idx == 0:
                expected_num_resp = len(systems_responses)

            n_sys_responses.append(len(systems_responses))

            list_sys_responses = ["-" for i in range(expected_num_resp)]

            for sys_resp_idx, each_sys_resp in enumerate(systems_responses):
                new_output_dataset_diffrow.append([row_idx, input_text, user_prompt, each_sys_resp])
                # if more than 'expected_num_resp', ignore
                if sys_resp_idx < expected_num_resp:
                    list_sys_responses[sys_resp_idx] = each_sys_resp

            new_output_dataset_diffcol.append([row_idx, input_text, user_prompt] + list_sys_responses)

        output_dataset = pd.DataFrame(new_output_dataset_diffrow, columns=[outputParaIdColumn, outputTextColumn, 'user_prompt', outputParaColumn])
        output_dataset.to_csv(outputFile + "_split_rows", sep="\t", index=False)

        if len(set(n_sys_responses)) != 1:
            print("Different number of output responses! : ", set(n_sys_responses))
            # ipdb.set_trace()

        output_dataset = pd.DataFrame(new_output_dataset_diffcol,
                                      columns=[outputParaIdColumn, outputTextColumn, 'user_prompt'] + [
                                          outputParaColumn + '_' + str(i + 1) for i in range(n_sys_responses[0])])
        output_dataset.to_csv(outputFile + "_split_cols", sep="\t", index=False)

        print("Output file name starts with: " + outputFile)

    # for row_idx, each_row in output_dataset_copy.iterrows():
    #
    #     if '"' in each_row[outputParaColumn]:
    #         #ipdb.set_trace()
    #         output_dataset_copy.at[row_idx, outputParaColumn] = re.sub('"', '', each_row[outputParaColumn])
    #
    #     if '\n' in each_row[outputParaColumn]:
    #         #ipdb.set_trace()
    #         output_dataset_copy.at[row_idx, outputParaColumn] = each_row[outputParaColumn].split('\n')[0]
    #
    # output_dataset_copy.to_csv(outputFile, sep="\t", index=False)
