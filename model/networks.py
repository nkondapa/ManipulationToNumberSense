import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models


class PrintLayer(nn.Module):
    def __init__(self, verbose=True):
        super(PrintLayer, self).__init__()
        self.verbose = verbose

    def forward(self, x):
        if self.verbose:
            print(x.shape)
        return x


class SiameseActionClassificationNet(nn.Module):
    def __init__(self, embedding_net, unique_id):
        super(SiameseActionClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.embedding_dimension = self.embedding_net.embedding_dimension
        self.name = 'siamese_classification_net-' + self.embedding_net.name
        self.unique_id = unique_id
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(self.embedding_dimension*2, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        cat_out = torch.cat([output1, output2], dim=1)
        out = self.nonlinear(cat_out)
        out = self.fc1(out)
        out = self.fc2(out)
        scores = F.log_softmax(out, dim=-1)
        return scores

    def get_embedding(self, x):
        return self.embedding_net(x)


class PreTrainedStackedEmbeddingNet(nn.Module):

    def __init__(self, embedding_dimension, pt_model='alex', model_params=None):
        super(PreTrainedStackedEmbeddingNet, self).__init__()
        self.name = '{pt}_stacked_embedding_net_{ed}'.format(pt=pt_model, ed=embedding_dimension)
        self.embedding_dimension = embedding_dimension

        if model_params is None:
            model_params = {}
        if pt_model == 'alex':
            full_pre_trained_model = models.alexnet(pretrained=True)
            nlayers = model_params.get('num_layers', 5)
            self.pre_trained_model = full_pre_trained_model.features[0:nlayers]
        else:
            raise Exception('Unknown pre-trained model')

        for param in self.pre_trained_model.parameters():
            param.requires_grad = False

        self.convnet = nn.Sequential(nn.Conv2d(192, 256, 5, padding=(2, 2)), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(256, 128, 3, padding=(1, 1)), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     )

        self.fc = nn.Sequential(nn.Linear(128 * 7 * 7, self.embedding_dimension),
                                nn.PReLU(),
                                nn.Linear(self.embedding_dimension, self.embedding_dimension),
                                )

    def forward(self, x):
        output = self.pre_trained_model.forward(x)
        output = self.convnet(output)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class UntrainedStackedEmbeddingNet(nn.Module):

    def __init__(self, embedding_dimension, pt_model='alex', model_params=None):
        super(UntrainedStackedEmbeddingNet, self).__init__()
        self.name = '{pt}_stacked_embedding_net_{ed}'.format(pt=pt_model, ed=embedding_dimension)
        self.embedding_dimension = embedding_dimension

        if model_params is None:
            model_params = {}
        if pt_model == 'alex':
            full_pre_trained_model = models.alexnet(pretrained=False)
            nlayers = model_params.get('num_layers', 5)
            self.pre_trained_model = full_pre_trained_model.features[0:nlayers]
        else:
            raise Exception('Unknown pre-trained model')

        # for param in self.pre_trained_model.parameters():
        #     param.requires_grad = False

        self.convnet = nn.Sequential(nn.Conv2d(192, 256, 5, padding=(2, 2)), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(256, 128, 3, padding=(1, 1)), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     )

        self.fc = nn.Sequential(nn.Linear(128 * 7 * 7, self.embedding_dimension),
                                nn.PReLU(),
                                nn.Linear(self.embedding_dimension, self.embedding_dimension),
                                )

    def forward(self, x):
        output = self.pre_trained_model.forward(x)
        output = self.convnet(output)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)