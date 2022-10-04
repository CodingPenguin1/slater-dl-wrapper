import torch


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_sizes=None, activations=None, dropout_probs=None, flatten=True):
        if hidden_sizes is None:
            hidden_sizes = []
        if activations is None:
            activations = ['ReLU']
        if dropout_probs is None:
            dropout_probs = [0]

        if type(activations) is str:
            activation_name = activations
            activations = [activation_name for _ in range(len(hidden_sizes))]
        if len(activations) == 1:
            activations *= len(hidden_sizes)
        assert len(activations) == len(hidden_sizes)

        if type(dropout_probs) is int:
            dropout_prob = dropout_probs
            dropout_probs = [dropout_prob for _ in range(len(hidden_sizes))]
        if len(dropout_probs) == 1:
            dropout_probs *= len(hidden_sizes)
        assert len(dropout_probs) == len(hidden_sizes)

        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        self.model = torch.nn.Sequential()
        if flatten:
            self.model.append(torch.nn.Flatten())
        for i in range(len(hidden_sizes)):
            # Layer
            if i == 0:
                self.model.append(torch.nn.Linear(num_features, hidden_sizes[i]))
            else:
                self.model.append(torch.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))

            # Dropout
            self.model.append(torch.nn.Dropout(dropout_probs[i]))

            # Activation
            if type(activations[i]) == str:
                self.model.append(getattr(torch.nn, activations[i])())
            else:
                self.model.append(activations[i])
        self.model.append(torch.nn.Linear(hidden_sizes[-1], num_classes))

    def forward(self, x):
        return self.model(x)
