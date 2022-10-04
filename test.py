from src.slater_dl_wrapper.models import MultiLayerPerceptron
import torch


if __name__ == '__main__':
    model = MultiLayerPerceptron(num_features=28**2, num_classes=10, hidden_sizes=[50, 20], activations=['ReLU'], flatten=True)
    print(model)
    print(dir(torch.nn.functional))
