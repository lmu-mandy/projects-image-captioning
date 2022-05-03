# This section of the project worked on by: Andrew Bruneel
# Code below trains our CNN --> RNN model on the caption and image database
# called "flikr8k", which was linked in the original github repository. The
# enitre set of data is around 1GB. This data is very similar to the COCO
# dataset in terms of the way pictures are framed and captioned (5 per image).
# Additionally, the cropping/transforming of images that are processed into the
# model is comparable as well.
# Original hyperparameters for this model are:
# embedding layers -- 256
# hidden layers -- 256
# dropout rate -- 50%
# learning rate -- 3e-4
# epochs -- 100
# Although these were altered multiple times throughout testing to see how the
# model was able to perform. This was mainly due to the limitations that were
# put on this model to run on GPU and still maintain performance without being
# able to load Google Colab checkpoint files. Because of this, performance
# had to be maximized per epoch for the (relatively) low amount of epochs used.

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from get_loader import get_loader
from model import CNNtoRNN
from nltk.translate.bleu_score import sentence_bleu


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Loading data from folders
    train_loader, dataset = get_loader(
        root_folder="data/images",
        annotation_file="data/captions.txt",
        transform=transform,
        num_workers=2,
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Deciding if the model will loaded between instances of running train.py
    load_model = True
    # Deciding if the model will be saved on the latest iterations
    save_model = True
    # Deciding if the CNN will be trained across epochs
    train_CNN = True

    # Hyperparameters (attempting to fine-tune for low amount of epochs)
    embed_size = 256
    hidden_size = 256
    # Constant vocab size/layers
    vocab_size = len(dataset.vocab)
    num_layers = 1
    # More hyperparameters
    learning_rate = 1e-4
    num_epochs = 1


    # for tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0

    # initialize model, loss etc
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Only finetune the CNN
    for name, param in model.encoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    if load_model:
        step = load_checkpoint(torch.load("/content/drive/MyDrive/NLP_Final_Project/my_checkpoint.pth"), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        # Storing bleu score
        # bleu_score = 0
        # all_bleu_scores = list()
        # Added clarity of epoch here, and working on BLEU Score as well
        print("Epoch #" + str(epoch))
        print("---------")
        # Uncomment the line below to see a couple of test cases
        print_examples(model, device, dataset)

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint, "/content/drive/MyDrive/NLP_Final_Project/my_checkpoint.pth")

        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            
            # Making a list to store captions used for bleu scores
            # bleu_captions = list()
            # Building list of captions split on spaces
            # for caption in captions[:-1]:
                # bleu_captions.append(caption.split((1, 1))
            # Calculating bleu score by each sentence
            # line_item_bleu_score = sentence_bleu(bleu_captions, outputs.shape[2])
            # all_bleu_scores.append(line_item_bleu_score)
            # Calculating BLEU score for the epoch
            # bleu_sum = sum(all_bleu_scores)
            # bleu_len = len(all_bleu_scores)
            # bleu_score = sum/len

            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()


if __name__ == "__main__":
    train()
