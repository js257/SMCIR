"""
AMIO -- All Model in One
"""
import torch.nn as nn

from .singleTask import *
from pytorch_transformers import BertConfig

class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.MODEL_MAP = {
            # single-task
            'smcir': SMCIR,
        }

        config = BertConfig.from_pretrained(args.pretrained, num_labels=1, finetuning_task='sst')
        self.Model = SMCIR.from_pretrained(args.pretrained, config=config, pos_tag_embedding=True,
                                               senti_embedding=True, polarity_embedding=True, args=args)

    def forward(self, text_x, audio_x, video_x, *args, **kwargs):
        return self.Model(text_x, audio_x, video_x, *args, **kwargs)
