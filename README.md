# Image-captioning
Image captioning using a CNN-LSTM architecture

References:
* https://www.kaggle.com/code/moatasemmohammed/image-captioning-with-flickr30k-dataset#Image-Captioning-with-Flickr30k-Dataset.
* https://www.kaggle.com/code/quadeer15sh/flickr8k-image-captioning-using-cnns-lstms#Image-Captioning

Two versions of the model were created and trained using [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset/data) dataset:
* V1 utilizing Resnet50
* V2 utilizing Resnet101, BN and dimentionality reduction. (Slightly better results)

## Results
Mean V1 BLEU-1 score: 0.7153
Mean V2 BlEU-1 score: 0.7511

Example test set image captioning results:
### V1

![download (8)](https://github.com/O-Okasha/Image-captioning/assets/57796344/41b1c3ab-908f-4c07-91c3-9c44e211faed)
![download (10)](https://github.com/O-Okasha/Image-captioning/assets/57796344/3146faef-fb7d-4af5-b57c-b42b923afc9a)

### V2

![download (9)](https://github.com/O-Okasha/Image-captioning/assets/57796344/13a7c527-5ad5-48f5-8f52-7ffdcf58670c)
![download (7)](https://github.com/O-Okasha/Image-captioning/assets/57796344/4c94842e-d47d-43bc-9aba-c9f09b630c15)

Example validation set image captioning results:

### V1

![download (1)](https://github.com/O-Okasha/Image-captioning/assets/57796344/1f8dba4b-36dc-40dc-98e7-a3f43d953e0b)

![download (2)](https://github.com/O-Okasha/Image-captioning/assets/57796344/fd8ef41d-1360-4fe7-8e8e-3cb72ecdd731)

### V2

![download (5)](https://github.com/O-Okasha/Image-captioning/assets/57796344/c9cc6bad-4aed-43a5-9b4c-e2a27de290be)

![download (4)](https://github.com/O-Okasha/Image-captioning/assets/57796344/825e2ea5-5d6c-4246-829d-1ef9257891c5)


