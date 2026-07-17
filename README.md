# [Sample-specific modality diagnosis and cross-modal enhancement for incomplete multimodal representations]([https://ieeexplore.ieee.org/document/10328787](https://ojs.aaai.org/index.php/AAAI/article/view/39102))
# Introduction
In multimodal sentiment analysis, modality missingness and quality degradation are common. Existing methods often rely on batch-level modality generation, generation but neglect sample-level missingness, hence their flexibility is limited severely in real-world scenarios. To address this, Sample-specific Modality Diagnosis and Cross-modal Enhancement for Incomplete Multimodal Representations (SMCIR) is proposed. Specifically, The Dynamic Multi-feature Fusion Detector (DMFD) is presented, which detects missingness and severity at the sample-level using indicators such as information entropy, modality similarity, and mutual information. Unlike batch-based methods, the DMFD provides fine-grained detection and adaptive responses, improving sensitivity to modality disturbances. Meanwhile, the Context-aware Modality Completion Generator (CMCG) is developed to restore missing modalities through context-guided reconstruction using multiscale feature fusion and cross-modal attention. In this way, the proposed CMCG method can avoid redundancy and inconsistency, enhancing the consistency and discriminativity of the fused representation. In CMCG, the text modality serves as a stable guide to improve context consistency. Experiments on the CMU-MOSI and CMU-MOSEI datasets show that SMCIR outperforms existing full-modal and non-recovery-based methods, well validating its efficacy and superiority in multimodal learning.
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

# If you use our SRCBTFusion-Net, please cite our paper:
<pre>
@inproceedings{chen2026sample,
  title={Sample-specific modality diagnosis and cross-modal enhancement for incomplete multimodal representations},
  author={Chen, Junsong and Liu, Jiyuan and Liu, Suyuan and Zhang, Wei and Li, Ao and Zhu, En and Liu, Xinwang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={24},
  pages={20154--20162},
  year={2026}
}
<pre>
