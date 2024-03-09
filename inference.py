import torch
from .utils import show_image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_cap(model, img, vocab):

    model.eval()
    with torch.no_grad():
        features = model.encoder(img[0:1].to(device))
        caps,alphas = model.decoder.generate_caption(features, vocab = vocab)
        caption = ' '.join(caps)
        show_image(img[0],title=caption)