import torch
import torch.nn as nn
from .Decoder import Decoder
from .Encoder import Encoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EncoderDecoder(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim)
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs