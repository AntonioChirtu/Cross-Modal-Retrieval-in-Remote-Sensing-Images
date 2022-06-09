import sys
import time

from matplotlib import pyplot as plt
from torchmetrics import Precision
from transformers import ViTFeatureExtractor
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import ToTensor, Normalize, Resize, Compose, RandomVerticalFlip, RandomHorizontalFlip, \
    RandomResizedCrop, RandomRotation

import torch
from torchvision import datasets, transforms
import torchvision.models as models

from tqdm import tqdm
from torch import nn

from sklearn.model_selection import GridSearchCV, KFold
import numpy as np

from PIL import Image

from img_model import ExtractFeatures, linearRegression, Sort

from scipy.ndimage import gaussian_filter

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from torchvision.models import vgg16, resnet18

import os
from data_loaders.data_loader import RemoteSensingDataset, TestDataset
import transformers

os.environ['TRANSFORMERS_OFFLINE'] = 'yes'

term_width = 10

TOTAL_BAR_LENGTH = 86.
last_time = time.time()
begin_time = last_time


class ViTFeatureExtractorTransforms:
    def __init__(self, feature_extractor):
        transform = []

        transform.append(Resize(224))

        transform.append(ToTensor())

        # transform.append(Normalize(0.5, 0.5))

        self.transform = Compose(transform)

    def __call__(self, x):
        return self.transform(x)


def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
  '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


if __name__ == '__main__':
    transformers.logging.set_verbosity_error()
    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    num_labels = 21
    batch_size = 16
    num_workers = 8
    max_epochs = 20
    split = 0.7
    sent_shape = 384 * 2
    final_dim = 384
    dataset_path = "UCM_70-30"
    full_path = "/home/antonio/PycharmProjects/BERT+ViT/" + dataset_path

    resnet18 = resnet18(pretrained=True)
    resnet18 = torch.nn.Sequential(*(list(resnet18.children())[:-1]))
    for param in resnet18.parameters():
        param.requires_grad = False
    # print(resnet18)

    # feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
    feature_extractor = resnet18

    transform_train = transforms.Compose([
        RandomRotation([-5, 5]),
        RandomVerticalFlip(0.3),
        RandomHorizontalFlip(0.3),
        ViTFeatureExtractorTransforms(feature_extractor),
        ExtractFeatures(feature_extractor),
    ])
    transform_test = transforms.Compose([
        ViTFeatureExtractorTransforms(feature_extractor),
        ExtractFeatures(feature_extractor),
    ])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    # model_bert = SentenceTransformer(
    #     '/home/antonio/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2')

    for param in model_bert.parameters():
        param.requires_grad = False

    train_dataset = RemoteSensingDataset(dataset_dir=full_path, type='train', split=split, transform=transform_train,
                                         model=model_bert, tokenizer=tokenizer)
    test_dataset = TestDataset(dataset_dir=full_path, type='test', split=split, transform=transform_test,
                               model=model_bert, tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=0)

    # model = linearRegression(sent_shape * 2, 2)
    model = linearRegression(final_dim * 2, 2)
    model.apply(reset_weights)
    for param in model.parameters():
        param.requires_grad = True

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    # ---------------- CLASSIC TRAINING ----------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    # For ViT
    # linear_visual = nn.Linear(18816*8, sent_shape)

    # For ResNet
    linear_visual = nn.Linear(512, final_dim)
    linear_sent = nn.Linear(sent_shape, final_dim)
    linear_sent.apply(reset_weights)
    linear_visual.apply(reset_weights)
    for param in linear_visual.parameters():
        param.requires_grad = False
    for param in linear_sent.parameters():
        param.requires_grad = False
    save_path_classic = '/home/antonio/PycharmProjects/BERT+ViT/BERT+ ' + 'ResNet' + '_net_CosSim' + dataset_path + '.pt'

    weights = [0.5, 0.5]
    class_weights = torch.FloatTensor(weights).cuda()
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # loss_function = nn.CosineEmbeddingLoss()
    loss_function = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    acc_list = []

    for epoch in range(max_epochs):
        model.train()
        print(f'Starting epoch {epoch + 1}')

        current_loss = 0.0
        correct = 0
        total = 0
        best_acc = 0.0
        train_correct = 0
        with torch.enable_grad():
            for i, data in enumerate(tqdm(train_loader), 0):
                images, sentences, targets = data

                images = torch.squeeze(images)
                image_features = linear_visual(images)
                sent_features = linear_sent(sentences)

                inputs = torch.cat([image_features, sent_features], dim=1)

                inputs = inputs.to(device)
                targets = targets.to(device)
                image_features = image_features.to(device)
                sent_features = sent_features.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = loss_function(outputs, targets)  # + 0.5 * (1 - cos(image_features, sent_features))
                # loss = torch.sum(loss)
                # loss = loss_function(image_features, sent_features, targets)

                train_correct += (torch.argmax(outputs) == targets).float().sum()

                loss.backward()
                optimizer.step()

                current_loss += loss.item()
                if i % 10 == 9:
                    print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 10))
                    current_loss = 0.0
            print('Accuracy of the network on the train images: %f %%' % (
                100 * train_correct / len(train_dataset)))

        # correct = 0
        # total = 0
        # sum = 0
        # top_k_preds = []
        # sentence = ""
        # with torch.no_grad():
        #     model.eval()
        #     for data in tqdm(test_loader):
        #         images, sentences, img_path, sentence = data
        #
        #         sentences = torch.squeeze(sentences)
        #         images = torch.squeeze(images)
        #
        #         image_features = linear_visual(images)
        #         sentences = linear_sent(sentences)
        #
        #         image_features = torch.unsqueeze(image_features, dim=0)
        #         sentences = torch.unsqueeze(sentences, dim=0)
        #         inputs = torch.cat([image_features, sentences], dim=1)
        #
        #         inputs = inputs.to(device)
        #
        #         outputs = model(inputs)
        #         predicted = torch.argmax(outputs)
        #
        #         top_k_preds.append([torch.max(outputs), img_path])
        #         sentence = sentence
        #
        #         # total += targets.size(0)
        #         # print("A: ", targets.size(0))
        #         # correct += (predicted == targets).sum().item()
        #
        #         sum += predicted
        #
        # top_3 = Sort(top_k_preds)
        # print(sentence)
        # print(top_3[:6])
        # print(top_3[-5:])
        # # for i, j in top_3:
        # #     im = Image.open(j[0])
        # #     im.show()
        # print('SUM is: ', sum)

        # if (100 * correct / total) > best_acc:
        #     best_acc = 100 * correct / total
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': current_loss,
        }, save_path_classic)

    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting testing')

    # Saving the model
    # save_path = f'./model-fold-{fold}.pth'
    # torch.save(model.state_dict(), save_path)

    checkpoint = torch.load(save_path_classic)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.eval()
    model.to(device)

    correct = 0
    total = 0
    sum = 0
    top_k_preds = {}
    sentence = ""
    idx = 0
    top_10_correct = 0
    with torch.no_grad():
        model.eval()
        for data in tqdm(test_loader):
            images, sentences, img_path, sentence, class_label = data

            sentences = torch.squeeze(sentences)
            images = torch.squeeze(images)

            image_features = linear_visual(images)
            sentences = linear_sent(sentences)

            image_features = torch.unsqueeze(image_features, dim=0)
            sentences = torch.unsqueeze(sentences, dim=0)
            inputs = torch.cat([image_features, sentences], dim=1)

            inputs = inputs.to(device)

            outputs = model(inputs)
            predicted = torch.argmax(outputs)

            # print(img_path[0].split('/')[-1])
            # print(outputs)
            # print(sentence)
            # print(torch.argmax(outputs))

            # top_k_preds.append([torch.max(outputs), img_path])
            top_k_preds[img_path[0].split('/')[-1][:-4]] = torch.max(outputs).cpu().detach().numpy()

            if idx % 629 == 0 and idx > 0:
                print(sentence)
                print(class_label)
                top_k_preds = sorted(top_k_preds.items(), key=lambda x: x[1], reverse=True)
                print(top_k_preds[:5])

                texts = [int(x[0]) // 100 for x in top_k_preds[:5]]

                if class_label.numpy() in texts:
                    top_10_correct += 1

                print(top_10_correct)

                top_k_preds = {}

            # total += targets.size(0)
            # print("A: ", targets.size(0))
            # correct += (predicted == targets).sum().item()

            idx += 1

            sum += predicted

