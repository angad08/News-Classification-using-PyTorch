import collections
import csv
from pathlib import Path
import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer
from torchvision.transforms import Compose, Resize, ToTensor, Normalize,FiveCrop


class TextDataset(torch.utils.data.Dataset):
  def __init__(self, fname, sentence_len):
    self.fname = pd.read_csv(fname)
    self.sentence_len = sentence_len
    self.text = self.fname.iloc[:,2].values
    self.label = self.fname.iloc[:,0].values - 1
    self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    # Tokenize the text
    self.output = self.tokenizer(self.text.tolist(),truncation=True,padding=True,max_length = self.sentence_len)
    self.input_ids = torch.tensor(self.output['input_ids'])

  def __len__(self):
    return len(self.fname)
 
  def __getitem__(self, idx): 
    self.text = self.input_ids[idx]
    self.labels = self.label[idx]
    return self.text,self.labels