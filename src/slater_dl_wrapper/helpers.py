from time import time
import torch


def train(model,
          num_epochs,
          train_loader,
          valid_loader,
          test_loader,
          optimizer,
          loss_func=torch.nn.functional.cross_entropy,
          device=torch.device('cpu')):

    start_time = time()
    minibatch_loss_list, train_acc_list, valid_acc_list, test_acc_list = [], [], [], []

    for epoch in range(num_epochs):
        # = TRAINING = #
        model.train()

        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)

            # Forward and back propogation
            logits = model(features)
            loss = loss_func(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # Update model parameters
            optimizer.step()

            # Logging
            minibatch_loss_list.append(loss.item())
            if not batch_idx % 50:
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} | Batch {batch_idx:04d}/{len(train_loader):04d} | Loss: {loss:.4f}')

        # = Evaluation = #
        model.eval()
        with torch.no_grad():  # save memory during inference
            train_acc = compute_accuracy(model, train_loader, device=device)
            valid_acc = compute_accuracy(model, valid_loader, device=device)
            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} | Train: {train_acc :.2f}% | Validation: {valid_acc :.2f}%')
            train_acc_list.append(train_acc.item())
            valid_acc_list.append(valid_acc.item())

        elapsed = (time() - start_time) / 60
        print(f'Time elapsed: {elapsed:.2f} min')

    elapsed = (time() - start_time) / 60
    print(f'Total Training Time: {elapsed:.2f} min')

    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f'Test accuracy {test_acc :.2f}%')

    return minibatch_loss_list, train_acc_list, valid_acc_list


def compute_accuracy(model, data_loader, device):

    with torch.no_grad():

        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100
