from models.CNNLSTM import EncoderDecoder
from .utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .Dataloader import get_loader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, criterion, vocab, optimizer, data_loader, num_epochs = 25, print_every = 100):
    
    vocab_size = len(vocab)

    for epoch in range(1,num_epochs+1):   
        for idx, (image, captions) in enumerate(iter(data_loader)):
            image, captions = image.to(device),captions.to(device)

            # Zero the gradients.
            optimizer.zero_grad()

            # Feed forward
            outputs,attentions = model(image, captions)

            # Calculate the batch loss.
            targets = captions[:,1:]
            loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
            
            # Backward pass.
            loss.backward()

            # Update the parameters in the optimizer.
            optimizer.step()

            if (idx+1)%print_every == 0:
                print("Epoch: {} loss: {:.5f}".format(epoch,loss.item()))
                
                
                #generate the caption
                model.eval()
                with torch.no_grad():
                    dataiter = iter(data_loader)
                    img,_ = next(dataiter)
                    features = model.encoder(img[0:1].to(device))
                    caps, alphas = model.decoder.generate_caption(features,vocab = vocab)
                    caption = ' '.join(caps)
                    show_image(img[0],title=caption)
                    
                model.train()
            
        #save the latest model
        save_model(model,epoch)
    return model, optimizer, loss

def main():
    imgs_path = ''
    caps_path = ''

    dataloader, dataset = get_loader(imgs_path, caps_path, transforms_list = None, BATCH_SIZE = 128, NUM_WORKER = 4)

    embed_size = 300
    vocab_size = len(dataset.vocab)
    attention_dim = 256
    encoder_dim = 2048
    decoder_dim = 512
    learning_rate = 3e-4

    model = EncoderDecoder(embed_size=embed_size, vocab_size = vocab_size,
                       attention_dim=attention_dim, encoder_dim=encoder_dim, decoder_dim=decoder_dim).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model, optimizer, loss = train(model= model, criterion = criterion, optimizer = optimizer, vocab = dataset.vocab,
                                   data_loader = dataloader, num_epochs= 25)
    


