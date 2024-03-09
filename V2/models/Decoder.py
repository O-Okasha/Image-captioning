import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super(Attention, self).__init__()
        
        self.attention_dim = attention_dim
        
        self.W1 = nn.Linear(decoder_dim,attention_dim)
        self.W2 = nn.Linear(encoder_dim,attention_dim)
        self.A = nn.Linear(attention_dim,1)
        
    def forward(self, features, hidden_state):
        
        w1_ah = self.W1(hidden_state)
        w2_hs = self.W2(features)
        
        combined_states = torch.tanh(w2_hs + w1_ah.unsqueeze(1))
        
        attention_scores = self.A(combined_states)
        attention_scores = attention_scores.squeeze(2)
        
        
        alpha = F.softmax(attention_scores, dim=1)
        
        attention_weights = features * alpha.unsqueeze(2)
        context_vector = attention_weights.sum(dim = 1)
        
        return alpha, context_vector
    
class Decoder(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()
        
        #save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.attention = Attention(encoder_dim,decoder_dim,attention_dim)
        
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim,decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        
        
        self.fcn = nn.Linear(decoder_dim,vocab_size)
        self.drop = nn.Dropout(drop_prob)
        
        
    
    def forward(self, features, captions):
        
        # Embed caption
        embeds = self.embedding(captions)
        
        # Initialize LSTM hidden states
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        

        seq_length = len(captions[0])-1 
        batch_size = captions.size(0)
        num_features = features.size(1)
        
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length,num_features).to(device)
                
        for s in range(seq_length):
            
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim = 1)
            h, c = self.lstm_cell(lstm_input, (h, c))
                    
            output = self.fcn(self.drop(h))
            
            preds[:,s] = output
            alphas[:,s] = alpha  
        
        
        return preds, alphas
    
    def generate_caption(self, features, max_len = 20, vocab = None):
 
        
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        alphas = []
        
        # Starting input
        word = torch.tensor(vocab.stoi['<BOS>']).view(1,-1).to(device)
        embeds = self.embedding(word)

        
        captions = []
        
        for i in range(max_len):
            alpha,context = self.attention(features, h)
            
            
            # Store the apla score
            alphas.append(alpha.cpu().detach().numpy())
            
            lstm_input = torch.cat((embeds[ : , 0], context), dim = 1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size, -1)
        
            
            # Select the word with highest probability
            predicted_word_idx = output.argmax(dim = 1)
            
            # Save the generated word
            captions.append(predicted_word_idx.item())
            
            # End if <EOS detected>
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break
            
            # Send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))
        
        # Map word index to word and return
        return [vocab.itos[idx] for idx in captions],alphas
    
    
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c