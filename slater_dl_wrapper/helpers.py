import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch


def _time_run(start):
    return time.strftime('%H:%M', time.gmtime(time.time() - start))


def train(model,
          num_epochs,
          train_loader,
          valid_loader=None,
          test_loader=None,
          optimizer='SGD',
          learning_rate=0.001,
          loss_func=torch.nn.functional.cross_entropy,
          device=torch.device('cpu')):

    # Setup
    if type(optimizer) == str:
        optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=learning_rate)
    assert issubclass(type(optimizer), torch.optim.Optimizer)
    model.to(device)

    # Table headers
    headers = ['Time', 'Epoch', 'Loss', 'Train']
    if valid_loader is not None:
        headers.append('Valid')
    if test_loader is not None:
        headers.append('Test')
    print('\t'.join(headers))

    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list, test_acc_list = [], [], [], []

    for epoch in range(num_epochs):
        # = TRAINING = #
        model.train()
        for features, targets in train_loader:  # batch loop
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            loss = loss_func(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            minibatch_loss_list.append(loss.item())

        # = Evaluation = #
        model.eval()
        with torch.no_grad():  # save memory during inference
            train_acc = compute_accuracy(model, train_loader, device=device)
            train_acc_list.append(train_acc)
            row_data = [_time_run(start_time), str(epoch + 1), f'{loss:.4f}', f'{train_acc:.2f}']

            if valid_loader is not None:
                valid_acc = compute_accuracy(model, valid_loader, device=device)
                valid_acc_list.append(valid_acc)
                row_data.append(f'{valid_acc:.2f}')

            if test_loader is not None:
                test_acc = compute_accuracy(model, test_loader, device=device)
                test_acc_list.append(test_acc)
                row_data.append(f'{test_acc:.2f}')
        print('\t'.join(row_data))

    if test_loader is None:
        return minibatch_loss_list, train_acc_list, valid_acc_list
    return minibatch_loss_list, train_acc_list, valid_acc_list, test_acc_list


def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0

        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()

    return (correct_pred.float() / num_examples * 100).item()


def plot_training_loss(minibatch_loss_list, num_epochs, iter_per_epoch, results_dir=None, averaging_iterations=100):
    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_loss_list)), (minibatch_loss_list), label='Minibatch Loss')

    if len(minibatch_loss_list) > 1000:
        ax1.set_ylim([0, np.max(minibatch_loss_list[1000:]) * 1.5])
    ax1.set_title('Minibatch Loss')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')

    ax1.plot(np.convolve(minibatch_loss_list,
                         np.ones(averaging_iterations,) / averaging_iterations,
                         mode='valid'),
             label='Running Average')
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs + 1))

    newpos = [e * iter_per_epoch for e in newlabel]
    tenth = int(len(newpos) / 10)

    ax2.set_xticks(newpos[::tenth])
    ax2.set_xticklabels(newlabel[::tenth])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()

    if results_dir is not None:
        image_path = os.path.join(results_dir, 'plot_training_loss.pdf')
        plt.savefig(image_path)
    plt.show()


def plot_accuracy(train_acc_list, valid_acc_list, results_dir):
    num_epochs = len(train_acc_list)

    plt.plot(np.arange(1, num_epochs + 1), train_acc_list, label='Training')
    plt.plot(np.arange(1, num_epochs + 1), valid_acc_list, label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    if results_dir is not None:
        image_path = os.path.join(
            results_dir, 'plot_acc_training_validation.pdf')
        plt.savefig(image_path)
    plt.show()
