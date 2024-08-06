#!/bin/bash

# a script to execute the following:
# 2. speech synthesis

#Step 0: set the execution environment
source /path_to/conda.sh
conda activate pytorch_1_6_clone  #pytorch_1_6


#Step 2: speech synthesis
#Inputs: ~ Input file with a text_column; output_directory to save synthesized speech
#Outputs: Input file
#         +++ an additional column for 'clean_utt_path'
# ---------------------------------------------------------------------------------
# UPDATE the following variables with correct values before execution

synthesisFile="utils/TTS_tacotron.py"

INPUT_DIR=$1
INPUT_FILE=$2
text_columns=$3 # a space separated list of columns; if numbers, starts with 1; else column name; it refers to the input to TTS
extra_columns=$4 # a space separated list of columns (numbers/names) for saving in output. eg: 'paraphrases_id'
outputFile="text2speech.txt"

inputFile=$INPUT_DIR/$INPUT_FILE #absolute path
date_ts=`date +"%F_%T"`

for text_col in $text_columns
do

outputFolder="$inputFile-TTS-$text_col"

#directory for storing wav files
mkdir $outputFolder
rm  $outputFolder/$outputFile

timeBefore=`date +%s`

sCount=-1
startTime=`date +"%s"`
echo "Start time : $startTime"


###### input a list of text to TTS script at a time ######
#python3 $synthesisFile -iFile $inputFile -oDir $outputFolder -tCol "$text_col" -eCols "$extra_columns" -hRow 0

python3 $synthesisFile -iFile $inputFile \
                       -oDir $outputFolder \
                       -tCol "$text_col" \
                       -eCols "$extra_columns" \
                       -oFile "$outputFolder/$outputFile" #with hRow None

###### input a single text to TTS script at a time ######
#while read line;
#do
#sCount=$(($sCount+1));
#p1=`echo "$line" | cut -f "$text_col"`
#fileName="utt_$sCount"
#python3 $synthesisFile "$p1" "$outputFolder/$fileName"".wav"
#echo -e "$p1\t$outputFolder/$fileName"".wav" >> $outputFolder/$outputFile   ## tab-delimited file
#done < $inputFile

echo "$inputFile - $outputFolder - $text_col"
timeAfter=`date +%s`
echo "Time taken for speech synthesis (s):" $(($timeAfter-$timeBefore))

done