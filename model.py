# This section of the project worked on by: Andrew Bruneel
# Code in this file generates both of the neural networks that are used by the
# model, and creates a link between them in the forward() method. There is a CNN
# used that has pre-training able to be turned off/on, and that is something I
# played with in the early stages of testing the model, to realize that it
# performed much better with them turned on. In addition, there is a Decoder RNN
# after the Encoder CNN. This uses an LSTM.

import torch
import torch.nn as nn
import statistics
import torchvision.models as models

# Creating the architecture for our encoderCNNN
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        # Setting if we will train our CNN over epochs or not
        self.train_CNN = train_CNN
        # Using a pretrained to improve initial performance
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        # Linear and RELU layers for structure
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.times = []
        # Using dropout of 50% to avoid overtraining and make model more generalizable
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        return self.dropout(self.relu(features))

# Decoder RNN to connect to our CNN
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        # Using an embedding layer, optimal for our vocab
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Connecting to an LSTM and linear layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        # Another instance of dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

# Linking CNN network to RNN here
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    # Creating captions for each passed in image
    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]
