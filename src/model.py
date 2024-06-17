import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import ResNet101_Weights as weights

import eca_resnet
import van
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-101 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(weights=weights.DEFAULT)
        # resnet = eca_resnet.eca_resnet101(pretrained=True)
        # resnet = van.van_b3(pretrained=True)
        # resnet.cuda()
        # checkpoint = torch.load('van_large_839.pth.tar') # ie, model_best.pth.tar
        # resnet.load_state_dict(checkpoint['state_dict']) # ckpt = torch.load('eca_net.pth.tar')
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
        # new_state_dict = OrderedDict()
        # for k, v in ckpt.items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v
        # # load params
        # resnet.load_state_dict(new_state_dict)
        # modules = list(resnet.children())[:-1]
        # #self.van = resnet
        # self.resnet = nn.Sequential(*modules)
        # self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        # self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        # print(self.resnet)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyperparameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # change for LSTM or RNN
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        weight = torch.load('../pretrain_weights.pt')
        weight = weight.to(device)
        self.prembed = nn.Embedding.from_pretrained(weight, freeze=True)

    def forward(self, features, captions, lengths, pretrained):
        """Decode image feature vectors and generates captions."""
        if pretrained:
            embeddings = self.prembed(captions).float()
        else:
            embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, pretrained, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            if pretrained:
                inputs = self.prembed(predicted).float()
            else:
                inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

    def stochastic_sample(self, features, temperature, pretrained, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)

            soft_out = F.softmax(outputs / temperature, dim=1)
            predicted = torch.multinomial(soft_out, 1).view(1)

            sampled_ids.append(predicted)
            if pretrained:
                inputs = self.prembed(predicted).float()
            else:
                inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
