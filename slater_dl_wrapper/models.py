import torch


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_sizes=None, activations=None, dropout_probs=None, flatten=True):
        '''
        Attributes:
            num_features (int): The number of features in the input data.
            num_classes (int): The number of classes in the output data.
            hidden_sizes (list): A list of integers representing the number of nodes in each hidden layer. The length of the list defines how many hidden layers there are.
            activations (list): A list of torch.nn.Module objects or strings representing the activation function for each hidden layer. The length of the list must be the same as the length of hidden_sizes.
            dropout_probs (list): A list of floats representing the dropout probability for each hidden layer. The length of the list must be the same as the length of hidden_sizes + 1, as the first element refers to dropout after the input layer. Dropout is applied after activation.
            flatten (bool): Whether or not to flatten the input data before passing it to the first hidden layer.
        '''

        # Default parameter values
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
            dropout_probs = [dropout_prob for _ in range(len(hidden_sizes) + 1)]
        if len(dropout_probs) == 1:
            dropout_probs *= len(hidden_sizes) + 1
        assert len(dropout_probs) == len(hidden_sizes) + 1

        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        # Input layer
        self.model = torch.nn.Sequential()
        if flatten:
            self.model.append(torch.nn.Flatten())
        self.model.append(torch.nn.Dropout(dropout_probs[0]))

        # Hidden layers
        for i in range(len(hidden_sizes)):
            # Connections
            if i == 0:
                self.model.append(torch.nn.Linear(num_features, hidden_sizes[i]))
            else:
                self.model.append(torch.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))

            # Activation
            if type(activations[i]) == str:
                self.model.append(getattr(torch.nn, activations[i])())
            else:
                self.model.append(activations[i])

            # Dropout
            self.model.append(torch.nn.Dropout(dropout_probs[i + 1]))

        self.model.append(torch.nn.Linear(hidden_sizes[-1], num_classes))

    def forward(self, x):
        return self.model(x)
