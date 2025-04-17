# Human Speech Perception in Noise: Can Large Language Models Paraphrase to Improve It?
### Anupama Chingacham, Miaoran Zhang, Vera Demberg, Dietrich Klakow

This repository consists of code and data created for our HuCLLM@ACL 2024 paper.

The work evaluates an LLM (like ChatGPT) on its ability to paraphrase a sentence, such that the generated paraphrase is acoustically more intelligible than the given input sentence, for human listeners in a noisy environment (eg., babble noise at SNR -5 dB).
The figure below depicts an overview of the two prompting approaches that we explored in this work.

![alt text](https://github.com/uds-lsv/llm_eval_PI-SPiN/blob/main/images/PI-SPiN_prompting_approaches.png)

---
 
Use the following steps for reproducing our evaluation results:


#### Standard Prompting
`` bash scripts/paraphrase_generation_zsl.sh ``

---
#### Prompt-and-Select
`` bash multi_step_exec.sh`` with ``scripts/paraphrase_generation_pas.sh`` in step 1.

---
#### Evaluate LLM

##### Automatic Evaluation
``bash ./get_para_metrics.sh``

##### Human Evaluation
Based the PWR-STOI of paraphrase pairs, two subsets of evaluation set is created.
 
1. Top 30 pairs: data/human_evaluation/top_30_pairs.txt
2. Random 30 pairs: data/human_evaluation/random_30_pairs.txt

---

Paraphrase to improve Speech Perception in Noise (PI-SPiN) is a text generation task, involving both textual attributes like semantic equivalence and non-textual attriutes like acoustic intelligibility. Prior studies used the following pipeline to identify acoustically intelligible paraphrase. 

![alt_text](https://github.com/uds-lsv/llm_eval_PI-SPiN/blob/main/images/PI-SPiN_pipeline.png)


