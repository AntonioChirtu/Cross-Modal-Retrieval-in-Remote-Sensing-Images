import warnings
from pathlib import Path
from typing import Callable, List, Optional

import pytorch_lightning as pl
import torchmetrics as metrics
import torch
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader
from transformers import (AdamW, get_linear_schedule_with_warmup, ViTForImageClassification, ViTFeatureExtractor)

from torch import nn
import numpy as np
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')


class ImageClassifier(pl.LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            learning_rate: float = 2e-5,
            adam_epsilon: float = 1e-8,
            weight_decay: float = 0.0,
            warmup_steps: int = 0,
            predictions_file: str = 'predictions.pt',
            num_labels: int = 10
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ViTForImageClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.accuracy_metric = metrics.Accuracy()

    def metric(self, preds, labels, mode='val'):
        a = self.accuracy_metric(preds, labels)
        return {f'{mode}_acc': a}

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        metric_dict = self.metric(preds, batch['labels'])
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_loss', val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        test_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        self.write_prediction('preds', preds, self.hparams.predictions_file)
        self.write_prediction('labels', batch['labels'], self.hparams.predictions_file)
        metric_dict = self.metric(preds, batch['labels'], mode='test')
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]

    @rank_zero_only
    def save_pretrained(self, save_dir):
        self.hparams.save_dir = save_dir
        self.model.save_pretrained(self.hparams.save_dir)
        # self.tokenizer.save_pretrained(self.hparams.save_dir)


class MLP_classic(nn.Module):

    def __init__(self, num_labels):
        super().__init__()
        self.layer1 = nn.Conv2d(1, 64, 3, stride=2)
        self.layer2 = nn.MaxPool2d(2)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.Conv2d(64, 32, 3, stride=2)
        self.layer5 = nn.MaxPool2d(2)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.Conv2d(32, 16, 3, stride=2)
        self.layer8 = nn.MaxPool2d(2)
        self.layer9 = nn.ReLU()
        self.layer10 = nn.Flatten()
        self.layer11 = nn.Linear(144, num_labels)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)

        return x


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear1 = torch.nn.Linear(inputSize, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = torch.nn.Linear(2048, 1024)
        self.relu2 = nn.ReLU()
        # self.linear3 = torch.nn.Linear(1024, 512)
        # self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.linear4 = torch.nn.Linear(1024, outputSize)
        self.sigmoid = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.relu1(x1)
        x3 = self.linear2(x2)
        x4 = self.relu2(x3)
        # x3 = self.linear3(x2)
        # x3 = self.relu3(x3)
        x5 = self.dropout(x4)
        out = self.linear4(x5)
        return self.sigmoid(out)


class ExtractFeatures(object):
    """Extract features of samples"""

    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, sample):
        # For ViT
        # features = self.feature_extractor(sample)
        # # output = np.concatenate((features['pixel_values'][0], sample.numpy()), axis=0)
        # return features['pixel_values'][0]

        # For ResNet
        features = self.feature_extractor(sample[None, ...])
        return features


def Sort(sub_li):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    sub_li.sort(key=lambda x: x[0], reverse=True)
    return sub_li
