import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
import torch

def visualization_and_evaluation(data, caption, actual):
    
    score = corpus_bleu([actual], [caption])
    
    fig = plt.figure(figsize=(14, 5))    
    ax = fig.add_subplot(1 , 2, 1, xticks=[], yticks=[])
    ax.imshow(data)

    ax = fig.add_subplot(1 , 2, 2)
    
    plt.axis('off')
    ax.plot()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 10)
    
    ax.text(0, 6, f'ACTUAL : {actual}',fontsize=10)
    ax.text(0, 5, f'PREDICT : {caption}',fontsize=10)
    ax.text(0, 4, f'BLEU SCORE : {score}',fontsize=10)
    
    plt.show()

def get_cap(cap):
    x = []
    ignore = ['<BOS>', '<EOS>', '<PAD>']
    for i in range(len(cap)):
        if cap[i] in ignore:
            continue
        else:
            x.append(cap[i])
    return ' '.join(x)

def show_image(img, title=None):

    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224 
    img[2] = img[2] * 0.225 
    img[0] += 0.485 
    img[1] += 0.456 
    img[2] += 0.406
    
    img = img.numpy().transpose((1, 2, 0))
    
    
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def save_model(model, num_epochs, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, loss, optimizer):
    model_state = {
        'num_epochs': num_epochs,
        'embed_size': embed_size,
        'vocab_size': vocab_size,
        'attention_dim':attention_dim,
        'encoder_dim':encoder_dim,
        'decoder_dim':decoder_dim,
        'state_dict':model.state_dict(),
        'loss' : loss,
        'optimizer_state': optimizer.state_dict()
    }

    torch.save(model_state,'cnnlstm.pth')