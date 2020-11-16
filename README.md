# A Time-Aware Transformer based model for Suicide Risk Assessment on Social Media
This codebase contains the python scripts for STATENet, the base model for the EMNLP20 paper [link](https://www.aclweb.org/anthology/2020.emnlp-main.619/).

## Environment & Installation Steps
Python 3.6 & Pytorch 1.5

```
pip install -r requirements.txt
```

## Run
Execute the following steps in the same environment:

```
cd STATENet_Time_Aware_Suicide_Assessment & python train.py --test
```

## Command Line Arguments
To run different variants of STATENet, perform ablation or tune hyperparameters, the following command-line arguments may be used:

1. -lr : learning rate. default : 0.001
2. -bs : batch size. default: 64
3. -e : number of epochs. default: 10
4. -hd : hidden layer dimension for TLSTM & LSTM. default : 100
5. -ed : dimensions of the embedded tweets. default : 768
6. -n : number of layers. default : 1
7. -d : dropout for all layers. default : 0.5
8. --base-model : choices={historic, historic-current, current} default: historic-current
9. --model : choices={tlstm", "bilstm", "bilstm-attention"} default : tlstm
10. --test : include to perform evaluation for test set
11. --data-dir : path to directory where the data folder is located. default: ""
12. --random : randomize tweet order for ablative comparison to empirically analyze the effect of temporal modeling.

## Dataset Format
Processed dataset format should be a DataFrame as a .pkl file having the following columns:

1. label : 0 or 1 for denoting a tweet as non-suicidal or suicidal respectively.
2. curr_enc : 768-dimensional encoding of the current tweet as a list. (STATENet uses SentenceBERT encodings for the current tweet)
3. enc : list of lists consisting of 768-dimensional encoding for each historical tweet. (STATENet uses BERT embeddings fine-tuned on EmoNe[1]) 
4. hist_dates : list containing the datetime objects corresponding to each historical tweet.

## Cite
If our work was helpful in your research, please kindly cite this work:
```
@inproceedings{sawhney2020time,
  title={A Time-Aware Transformer Based Model for Suicide Ideation Detection on Social Media},
  author={Sawhney, Ramit and Joshi, Harshit and Gandhi, Saumya and Shah, Rajiv Ratn},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2020}
}
```

## Ethical Considerations
The preponderance of the work presented in our discussion presents heightened ethical challenges. 
As explored in \citet{Coppersmith2018}, we address the trade-off between privacy and effectiveness. 
While data is essential in making models like STATENet effective, we must work within the purview of acceptable privacy practices to avoid coercion and intrusive treatment. 
We believe that intervention is a critical step, and STATENet should be used in conjunction with clinical professionals.
To that end, we utilize publicly available Twitter data in a purely observational, and non-intrusive manner.
All tweets shown as examples in our paper and example data have been paraphrased as per the moderate disguise scheme suggested in [4] to protect the privacy of individuals, and attempts should not be made to reverse identify individuals. 
Assessments made by STATENet are sensitive and should be shared selectively to avoid misuse, such as Samaritan's Radar.
Our work does not make any diagnostic claims related to suicide. 
We study the social media posts in a purely observational capacity and do not intervene with the user experience in any way.


### Note on data
In this work we utilize data from prior work [1, 2].
In compliance with Twitter's privacy guidelines, and the ethical considerations discussed in prior work [2] on suicide ideation detection on social media data, we redirect researchers to theprior work that introduced Emonet [1] and the suicide ideation Twitter dataset [2] to request access to the data.

Please follow the below steps to preprocess the data before feeding it to STATENet:
1. Obtain tweets from Emonet [1], or any other (emotion-based) dataset, to fine-tune a pretrained transformer model (we used BERT-base-cased; English). For Emonet, the authors share the tweet IDs in their dataset (complying to Twitter's privacy guidelines). These tweets then have to be hydrated for further processing.
2. Alternatively, any existing transformer can be used.
3. Using this pretrained transformer, encode all *historical* tweets to obtain a embeddings per historical tweet.
4. For the tweets to be assessed (for which we want to assess suicidal risk), encode the tweets using a pretrained encoder (We use SentenceBERT [3]) to obtain an embedding per tweet to be assessed.
the data provided is a small sample of the original dataset and hence the results obtained on this sample are not fully representative of the results that are obtained on the full dataset.
5. Using these embeddings, create a dataset file in the format explained above under the data directory.
6. We provide the sample format in data/samp_data.pkl

### References

[1] Abdul-Mageed, Muhammad, and Lyle Ungar. "Emonet: Fine-grained emotion detection with gated recurrent neural networks." Proceedings of the 55th annual meeting of the association for computational linguistics (volume 1: Long papers). 2017.

[2] Sawhney, Ramit, Prachi Manchanda, Raj Singh, and Swati Aggarwal. "A computational approach to feature extraction for identification of suicidal ideation in tweets." In Proceedings of ACL 2018, Student Research Workshop, pp. 91-98. 2018.

[3] Reimers, Nils, and Iryna Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 3973-3983. 2019.

[4] Bruckman, A., 2002. Studying the amateur artist: A perspective on disguising data collected in human subjects research on the Internet. Ethics and Information Technology, 4(3), pp.217-231
