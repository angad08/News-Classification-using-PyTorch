import torch
import torch.nn as nn
import torchvision
from transformers import AutoModelForSequenceClassification,DistilBertModel

class TextMLP(nn.Module):
    # It should be a simple MLP with 8 Linear layers. It should first embed the inputs into a vocabulary of size 30522.
    # Use an output feature size of 256 in all hidden layers and a feature size of 128 for the embeddings.
    # Flatten the sentence after embedding, but before it goes into any Linear layers.
    # Use batch norm and ReLU.
    # Train for 1000 epochs with learning rate of 0.001 and a batch size of 512.
    def __init__(self, vocab_size, sentence_len, hidden_size, n_classes=4):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Embedding(vocab_size, hidden_size//2),
            nn.Flatten(),
            nn.BatchNorm1d(sentence_len*hidden_size//2),
            nn.Linear(sentence_len*hidden_size//2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes)
        )
    def forward(self, x):
        return self.seq(x)