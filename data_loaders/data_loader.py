import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import json
import random

os.environ['TRANSFORMERS_OFFLINE'] = 'yes'


class RemoteSensingDataset(Dataset):
    def __init__(self, dataset_dir, type, split, model, tokenizer=None, transform=None):
        self.img_labels = []
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.images = []
        self.type = type
        self.split = split
        self.tokenizer = tokenizer
        self.model = model

        # JSON file containing information regarding the link between images and sentences
        self.json = json.load(open(os.path.join(dataset_dir, 'dataset.json')))

        if type == 'train':
            for class_dir in os.listdir(os.path.join(self.dataset_dir, 'train')):
                train_files = os.listdir(os.path.join(self.dataset_dir, 'train', class_dir))
                for idx, img in enumerate(train_files):
                    self.img_labels.append(int(class_dir))
                    self.images.append(img)
        else:
            for class_dir in os.listdir(os.path.join(self.dataset_dir, 'test')):
                test_files = os.listdir(os.path.join(self.dataset_dir, 'test', class_dir))
                test_files = sorted(test_files)
                for idx, img in enumerate(test_files):
                    self.img_labels.append(int(class_dir))
                    self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        idx = idx % len(self.images)

        label = 1
        # if idx % 5 == 0:
        #     label = 0

        if label == 1:
            img_path = os.path.join(self.dataset_dir, self.type, str(self.img_labels[idx]), self.images[idx])
            sentences = [x['raw'] for x in self.json['images'][int(self.images[idx].split('.')[0])]['sentences']]
        else:
            l = list(set(self.img_labels))
            l.remove(self.img_labels[idx])
            new_label = random.choice(l)
            new_img = self.images[idx].split('.')[0][-2:]
            if (int(new_label)) == 0:
                new_img = str(int(new_img)) + '.tif'
            else:
                if int(new_img) < 10:
                    new_img = '0' + str(int(new_img))
                new_img = str(int(new_label)) + new_img + '.tif'
            if self.type == 'test' and new_img == '0.tif':
                new_img = '8.tif'
            img_path = os.path.join(self.dataset_dir, self.type, str(new_label), new_img)
            sentences = [x['raw'] for x in self.json['images'][int(self.images[idx].split('.')[0])]['sentences']]

        sentences = sentences[0]

        # sentence_embeddings = self.model.encode(sentences)

        # parcurgi setul de date, salvezi trasaturile de la imagine si text intr-un npy, dupa load
        encoded_input = self.tokenizer(sentences, return_tensors='pt')
        word_embeddings = self.model(**encoded_input)
        word_embeddings = torch.squeeze(word_embeddings['last_hidden_state'])
        sentence_embeddings = torch.mean(word_embeddings, dim=0)

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        sentence_embeddings = (sentence_embeddings - torch.min(sentence_embeddings)) / (
                torch.max(sentence_embeddings) - torch.min(sentence_embeddings))

        image -= image.min(1, keepdim=True)[0]
        image /= image.max(1, keepdim=True)[0]

        return image, sentence_embeddings, label


class TripletLossDataset(Dataset):
    def __init__(self, dataset_dir, type, split, model, tokenizer=None, transform=None):
        self.img_labels = []
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.images = []
        self.type = type
        self.split = split
        self.tokenizer = tokenizer
        self.model = model

        # JSON file containing information regarding the link between images and sentences
        self.json = json.load(open(os.path.join(dataset_dir, 'dataset.json')))

        if type == 'train':
            for class_dir in os.listdir(os.path.join(self.dataset_dir, 'train')):
                train_files = os.listdir(os.path.join(self.dataset_dir, 'train', class_dir))
                for idx, img in enumerate(train_files):
                    self.img_labels.append(int(class_dir))
                    self.images.append(img)
        else:
            for class_dir in os.listdir(os.path.join(self.dataset_dir, 'test')):
                test_files = os.listdir(os.path.join(self.dataset_dir, 'test', class_dir))
                test_files = sorted(test_files)
                for idx, img in enumerate(test_files):
                    self.img_labels.append(int(class_dir))
                    self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        idx = idx % len(self.images)

        img_path = os.path.join(self.dataset_dir, self.type, str(self.img_labels[idx]), self.images[idx])
        sentences = [x['raw'] for x in self.json['images'][int(self.images[idx].split('.')[0])]['sentences']]

        l = list(set(self.img_labels))
        l.remove(self.img_labels[idx])
        new_label = random.choice(l)
        new_img = self.images[idx].split('.')[0][-2:]
        if (int(new_label)) == 0:
            new_img = str(int(new_img)) + '.tif'
        else:
            if int(new_img) < 10:
                new_img = '0' + str(int(new_img))
            new_img = str(int(new_label)) + new_img + '.tif'
        if self.type == 'test' and new_img == '0.tif':
            new_img = '8.tif'
        wrong_img_path = os.path.join(self.dataset_dir, self.type, str(new_label), new_img)
        sentences = sentences[0]

        # sentence_embeddings = self.model.encode(sentences)

        # parcurgi setul de date, salvezi trasaturile de la imagine si text intr-un npy, dupa load
        encoded_input = self.tokenizer(sentences, return_tensors='pt')
        word_embeddings = self.model(**encoded_input)
        word_embeddings = torch.squeeze(word_embeddings['last_hidden_state'])
        sentence_embeddings = torch.mean(word_embeddings, dim=0)

        image = Image.open(img_path)
        wrong_image = Image.open(wrong_img_path)

        if self.transform:
            image = self.transform(image)

        sentence_embeddings = (sentence_embeddings - torch.min(sentence_embeddings)) / (
                torch.max(sentence_embeddings) - torch.min(sentence_embeddings))

        image -= image.min(1, keepdim=True)[0]
        image /= image.max(1, keepdim=True)[0]

        wrong_image -= wrong_image.min(1, keepdim=True)[0]
        wrong_image /= wrong_image.max(1, keepdim=True)[0]

        return image, sentence_embeddings, wrong_image


class TestDataset(Dataset):
    def __init__(self, dataset_dir, type, split, model, tokenizer=None, transform=None):
        self.img_labels = []
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.images = []
        self.type = type
        self.split = split
        self.tokenizer = tokenizer
        self.model = model

        self.json = json.load(open(os.path.join(dataset_dir, 'dataset.json')))

        for class_dir in os.listdir(os.path.join(self.dataset_dir, 'test')):
            test_files = os.listdir(os.path.join(self.dataset_dir, 'test', class_dir))
            test_files = sorted(test_files)
            for idx, img in enumerate(test_files):
                self.img_labels.append(int(class_dir))
                self.images.append(img)

        self.idx = 0

    def __len__(self):
        return len(self.images)**2 - len(self.images)

    def __getitem__(self, idx):
        new_idx = self.idx

        if idx % (len(self.images)) == 0 and idx > 0:
            self.idx += 1

        idx = idx % len(self.images)

        # print(self.json['images'][int(self.images[new_idx].split('.')[0])])
        sentences = [x['raw'] for x in self.json['images'][int(self.images[new_idx].split('.')[0])]['sentences']]
        sent_id = [x for x in self.json['images'][int(self.images[new_idx].split('.')[0])]['sentids']]
        sent_id = sent_id[0]
        # print(sent_id)
        label = sent_id / 5 // 100
        # print(label)
        sentences = sentences[0]

        raw_sent = sentences

        encoded_input = self.tokenizer(sentences, return_tensors='pt')
        word_embeddings = self.model(**encoded_input)
        word_embeddings = torch.squeeze(word_embeddings['last_hidden_state'])
        sentence_embeddings = torch.mean(word_embeddings, dim=0)

        img_path = os.path.join(self.dataset_dir, self.type, str(self.img_labels[idx]), self.images[idx])

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        sentence_embeddings = (sentence_embeddings - torch.min(sentence_embeddings)) / (
                torch.max(sentence_embeddings) - torch.min(sentence_embeddings))

        image -= image.min(1, keepdim=True)[0]
        image /= image.max(1, keepdim=True)[0]

        return image, sentence_embeddings, img_path, raw_sent, label


class TestDatasetSingle(Dataset):
    def __init__(self, dataset_dir, type, split, model, tokenizer=None, transform=None):
        self.img_labels = []
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.images = []
        self.type = type
        self.split = split
        self.tokenizer = tokenizer
        self.model = model

        self.json = json.load(open(os.path.join(dataset_dir, 'dataset.json')))

        for class_dir in os.listdir(os.path.join(self.dataset_dir, 'test')):
            test_files = os.listdir(os.path.join(self.dataset_dir, 'test', class_dir))
            test_files = sorted(test_files)
            for idx, img in enumerate(test_files):
                self.img_labels.append(int(class_dir))
                self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        new_idx = 10

        # print(self.json['images'][int(self.images[new_idx].split('.')[0])])
        sentences = [x['raw'] for x in self.json['images'][int(self.images[new_idx].split('.')[0])]['sentences']]
        sent_id = [x for x in self.json['images'][int(self.images[new_idx].split('.')[0])]['sentids']]
        sent_id = sent_id[0]
        label = sent_id / 5 // 100
        sentences = sentences[0]

        raw_sent = sentences

        encoded_input = self.tokenizer(sentences, return_tensors='pt')
        word_embeddings = self.model(**encoded_input)
        word_embeddings = torch.squeeze(word_embeddings['last_hidden_state'])
        sentence_embeddings = torch.mean(word_embeddings, dim=0)

        img_path = os.path.join(self.dataset_dir, self.type, str(self.img_labels[idx]), self.images[idx])

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        sentence_embeddings = (sentence_embeddings - torch.min(sentence_embeddings)) / (
                torch.max(sentence_embeddings) - torch.min(sentence_embeddings))

        image -= image.min(1, keepdim=True)[0]
        image /= image.max(1, keepdim=True)[0]

        return image, sentence_embeddings, img_path, raw_sent, label
