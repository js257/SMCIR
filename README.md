# Prepare
You need to modify the dataset path and the pretrained language model path in config_regression.json to your own. If you don't intend to train the model, you can load the pre-trained weights we provide. In this case, you also need to modify the weight download URL in run.py and comment out the training-related code.
## Dataset
Download the MOSI and MOSEI pkl file [https://drive.google.com/drive/folders/1_u1Vt0_4g0RLoQbdslBwAdMslEdW1avI?usp=sharing](https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk). Put it under the "./dataset" directory.

## Pre-trained language model
Download the Bert language model files [https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip).

## Model weight
If you do not want to train the model, you can load the pre-trained model weights. The training weights obtained on MOSI and MOSEI are available at the following link: [https://drive.google.com/drive/folders/1hJ8cD_piHwOGe5FY8s5BmOvx8PnaqPlo?usp=drive_link](https://drive.google.com/drive/folders/1eV1a-iXE0neZ_ejNccLQ6Keu8bN6VC1v).

# Run
'''
python test.py
'''

Note: To get results close to those in our paper, you can set the seed in args to 1111. The experimental results of this paper are obtained on the Linux system.

