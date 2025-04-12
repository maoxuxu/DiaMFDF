# DiaMFDF

## Requirements

The model is implemented using PyTorch. The versions of the main packages:

+ python                    3.7.9                
+ torch                     1.9.0+cu111 
+ torchaudio                0.9.0      
+ torchvision               0.10.0+cu111  
+ transformers              4.20.1    
+ numpy                     1.21.6

If you are using a different version of the environment, you can try using parameter search to achieve the best score.

Install the other required packages:
``` bash
pip install -r requirements.txt
```


## TRAIN
### 1. Get Dataset
prepare data, dependency parse by spacy. You can download it from ```https://drive.google.com/drive/folders/1OJCOh6lyUny_5PkXq_qEUtHVf46PGqkL?usp=drive_link``` to ```data/dataset```;
OR
Download origin dataset from ```https://github.com/unikcc/DiaASQ/tree/master/data/dataset``` to data/dataset;

then install Spacy and download the model weight:
  - 'zh_core_web_trf' for ZH dataset
  - 'en_core_web_trf' for EN dataset

and you can get the dependency parse data by running the following code:
```
cd ./data/dataset/
python gen_dep_dataset_by_spacy.py
```

### 2. Training
```
bash scrip/train.sh
```
