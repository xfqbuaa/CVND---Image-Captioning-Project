import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        # build model
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        #print("Features:", features.shape)
        #print("Captions:", captions.shape)
        embeddings = self.embed(captions[:,:-1])
        #[batch_size, captions.shape[1]-1, embed_size]
        #print("Embeddings:", embeddings.shape)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        #print("Cat:", embeddings.shape)
        #[batch_size, captions.shape[1], embed_size]
        hiddens, _ = self.lstm(embeddings)
        #print("Hiddens:", hiddens.shape)
        #[batch_size, captions.shape[1], hidden_size]
        outputs = self.linear(hiddens)
        #print("Outputs:",outputs.shape)
        #[batch_size, captions.shape[1], vocab_size]
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_ids = []
        # greedy search
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)           # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                    # predicted: (batch_size)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)                    # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                     # inputs: (batch_size, 1, embed_size)

        return sampled_ids