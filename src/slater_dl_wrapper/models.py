import torch


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_sizes=None, activations=None, flatten=True):
        # Default kwarg values
        if hidden_sizes is None:
            hidden_sizes = []
        if activations is None:
            activations = []

        if type(activations) is str:
            activation_name = activations
            activations = [activation_name for _ in range(len(hidden_sizes))]
        if len(activations) == 1:
            activations *= len(hidden_sizes)
        assert len(activations) == len(hidden_sizes)

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
            elif i == len(hidden_sizes) - 1:
                self.model.append(torch.nn.Linear(hidden_sizes[i - 1], num_classes))
            else:
                self.model.append(torch.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))

            # Activation
            if type(activations[i]) == str:
                self.model.append(getattr(torch.nn, activations[i])())
            else:
                self.model.append(activations[i])

    def forward(self, x):
        return self.model(x)
