Authors:
Dongqi Pu(2687343),Wenxin Chen(2610585),Siddharth Chaubal(2698570)

1.For PartA, you can run PartA.py (in the Analyzing Language folder) directly to get the corresponding answer. 

It should be noted that for the POS-Tagging task (subtask of this part), the code will output a 'pos_tag_token_frequencies_with_tokens.txt' file, which contains all the Finegrained-Universal-token-frequency combinations in reverse order. Therefore, you can get the top three most frequent tokens by searching.

2.Run PartB.py (in the Classifying Language folder) directly to get the corresponding answer of PartB.

3.For PartC (in the Representing Language folder), it will use the'bert-base-uncased' pre-training model. If you load it for the first time, the API interface provided by Huggingface will automatically download it for you, so don't worry. The pre-training code is provided in two ways, in the form of a script file (PartC.py) and a notebook (PartC.ipynb). You can choose one of them to get the fine-tuned BERT model. Please note that the fine-tuned model will be stored in the folder './fineturned_bert/'. You may have to note that for the selection of comparing different hyperparameters (please view the submitted PDF assignment document), please change the parameter value of the corresponding parameter in the source code to get the comparison result. If you want to use Google colab to run PartC.ipynb, please do not forget to upload the dataset file in "./Representing Language/dataset/" to colab.

For the second half of PartC, compare the sentence similarity tasks, please save the fine-tuned model to the 'fineturned_bert' folder, and then you can run the untuned_BERT_similarity.py file and the finetuned_BERT_similarity.py file respectively to get the corresponding answer.