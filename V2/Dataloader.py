import pandas as pd
from PIL import Image
import numpy as np
import spacy
import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T
import torch
import re
spacy_eng = spacy.load("en_core_web_sm")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Vocabulary:
    def __init__(self):
        
        self.itos = {0 : "<PAD>", 1 : "<BOS>", 2 : "<EOS>", 3 : ' ', 4: 'ERR'}
        self.stoi = {v:k for k,v in self.itos.items()}
        self.vocab = set()
    def __len__(self): 
        return len(self.itos)

    def tokenize(self, text):
        return [token.text.strip().lower() for token in spacy_eng.tokenizer(text)]
    
    def build_vocab(self, sentence_list):
        idx = 5
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                
                if word not in self.vocab:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    self.vocab.add(word)
                    idx += 1
    
    def numericalize(self,text):
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else 4 for token in tokenized_text ] 
    
class FlickrDataset(Dataset):

    def __init__(self, root_dir, caption_file, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file, sep='|')
        self.transform = transform
        self.df = self.df.dropna().reset_index(drop=True)
        self.preprocess(self.df[' comment'].values)
        
        self.vocab = Vocabulary()
        self.vocab.build_vocab(self.df[' comment'].values)
        
    def preprocess(self, data):
        for i in range(len(data)):
            data[i] = data[i].lower()
            strip_chars = "[!\"#$%&'()*+,-./:;=?@[\]^_`{|}~1234567890]"
            data[i] = re.sub(strip_chars, "", data[i])
            data[i] = data[i].replace('  ', ' ')
            data[i] = data[i].strip()
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        caption = self.df[' comment'][idx]
        img_name = self.df['image_name'][idx]
        img_location = os.path.join(self.root_dir,img_name)
        img = Image.open(img_location).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
        
        caption_vec = []
        caption_vec += [self.vocab.stoi["<BOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]
        
        return img, torch.tensor(caption_vec)
    
class CapsCollate:

    def __init__(self, pad_idx, batch_first = False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
    
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first = self.batch_first, padding_value = self.pad_idx)
        return imgs,targets
    
def get_loader(data_location, caps_location, transforms_list = None, BATCH_SIZE = 128, NUM_WORKER = 4):

    if transforms_list == None:
        transforms_list = [T.Resize(226), T.RandomCrop(224), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    transforms = T.Compose(transforms_list)
    dataset =  FlickrDataset(root_dir = data_location, caption_file = caps_location, transform=transforms)

    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True,
                         collate_fn=CapsCollate(pad_idx = 0, batch_first = True))
    return data_loader, dataset