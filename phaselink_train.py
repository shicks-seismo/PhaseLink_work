#!/usr/bin/python
#-----------------------------------------------------------------------------------------------------------------------------------------

# PhaseLink: Earthquake phase association with deep learning
# Author: Zachary E. Ross
# Seismological Laboratory
# California Institute of Technology

# Script Description:
# Script to train a stacked bidirectional GRU model to link phases together. This code takes the synthetic training dataset produced using p
# haselink_dataset and trains a deep neural network to associate individual phases into events.

# Usage:
# python phaselink_train.py config_json
# For example: python phaselink_train.py params.json

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import numpy as np
import os
import torch
import torch.utils.data
import sys
import json
import pickle
import glob
import gc 
import matplotlib.pyplot as plt 
from torch.utils.data.sampler import SubsetRandomSampler

#----------------------------------------------- Define main functions -----------------------------------------------
class MyDataset(torch.utils.data.Dataset):
    """Function to preprocess a dataset into the format required by 
    pytorch for training."""
    def __init__(self, data, target, device, transform=None):
        self.data = torch.from_numpy(data).float().to(device)
        self.target = torch.from_numpy(target).short().to(device)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

class StackedGRU(torch.nn.Module):
    """Class defining the stacked bidirectional GRU network."""
    def __init__(self):
        super(StackedGRU, self).__init__()
        self.hidden_size = 128
        self.fc1 = torch.nn.Linear(5, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 32)
        self.fc4 = torch.nn.Linear(32, 32)
        self.fc5 = torch.nn.Linear(32, 32)
        self.fc6 = torch.nn.Linear(2*self.hidden_size, 1)
        self.gru1 = torch.nn.GRU(32, self.hidden_size, \
            batch_first=True, bidirectional=True)
        self.gru2 = torch.nn.GRU(self.hidden_size*2, self.hidden_size, \
            batch_first=True, bidirectional=True)

    def forward(self, inp):
        out = self.fc1(inp)
        out = torch.nn.functional.relu(out)
        out = self.fc2(out)
        out = torch.nn.functional.relu(out)
        out = self.fc3(out)
        out = torch.nn.functional.relu(out)
        out = self.fc4(out)
        out = torch.nn.functional.relu(out)
        out = self.fc5(out)
        out = torch.nn.functional.relu(out)
        out = self.gru1(out)
        h_t = out[0]
        out = self.gru2(h_t)
        h_t = out[0]
        out = self.fc6(h_t)
        #out = torch.sigmoid(out)
        return out

class Model():
    """Class to create and train a bidirectional GRU model."""
    def __init__(self, network, optimizer, model_path):
        self.network = network
        self.optimizer = optimizer
        self.model_path = model_path

    def train(self, train_loader, val_loader, n_epochs, enable_amp=False):
        """Function to perform the training of a bidirectional GRU model.
        Loads and trains the data."""
        from torch.autograd import Variable
        import time
        if enable_amp:
            import apex.amp as amp

        #pos_weight = torch.ones([1]).to(device)*24.264966334432359
        #loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = torch.nn.BCEWithLogitsLoss()
        #loss = torch.nn.BCELoss()
        n_batches = len(train_loader)
        training_start_time = time.time()

        for epoch in range(n_epochs):
            running_loss = 0.0
            running_acc = 0
            running_val_acc = 0
            print_every = n_batches // 10
            start_time = time.time()
            total_train_loss = 0
            total_val_loss = 0
            total_val_acc = 0
            running_sample_count = 0

            for i, data in enumerate(train_loader, 0):
                # Get inputs/outputs and wrap in variable object
                inputs, labels = data
                #inputs = inputs.to(device)
                #labels = labels.to(device)

                # Set gradients for all parameters to zero
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.network(inputs)

                # Backward pass
                outputs = outputs.view(-1)

                labels = labels.view(-1)

                if enable_amp:
                    loss_ = loss(outputs, labels.float())
                    with amp.scale_loss(loss_, self.optimizer) as loss_value:
                        loss_value.backward()
                else:
                    loss_value = loss(outputs, labels.float())
                    loss_value.backward()

                # Update parameters
                self.optimizer.step()

                with torch.no_grad():
                    # Print statistics
                    running_loss += loss_value.data.item()
                    total_train_loss += loss_value.data.item()

                    # Calculate categorical accuracy
                    pred = torch.round(torch.sigmoid(outputs)).short()

                    running_acc += (pred == labels).sum().item()
                    running_sample_count += len(labels)

                    # Print every 10th batch of an epoch
                    if (i + 1) % (print_every + 1) == 0:
                        print("Epoch {}, {:d}% \t train_loss: {:.4e} "
                            "train_acc: {:4.2f}% took: {:.2f}s".format(
                            epoch + 1, int(100 * (i + 1) / n_batches),
                            running_loss / print_every,
                            100*running_acc / running_sample_count,
                            time.time() - start_time))

                        # Reset running loss and time
                        running_loss = 0.0
                        start_time = time.time()

            running_sample_count = 0
            y_pred_all, y_true_all = [], []

            prec_0 = 0
            prec_n_0 = 0
            prec_1 = 0
            prec_n_1 = 0
            reca_0 = 0
            reca_n_0 = 0
            reca_1 = 0
            reca_n_1 = 0
            pick_precision = 0
            pick_recall = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    # Wrap tensors in Variables
                    #inputs = inputs.to(device)
                    #labels = labels.to(device)

                    # Forward pass only
                    val_outputs = self.network(inputs)
                    val_outputs = val_outputs.view(-1)
                    labels = labels.view(-1)
                    val_loss = loss(val_outputs, labels.float())
                    total_val_loss += val_loss.data.item()

                    # Calculate categorical accuracy
                    y_pred = torch.round(torch.sigmoid(val_outputs)).short()
                    running_val_acc += (y_pred == labels).sum().item()
                    running_sample_count += len(labels)

                    #y_pred_all.append(pred.cpu().numpy().flatten())
                    #y_true_all.append(labels.cpu().numpy().flatten())

                    y_true = labels

                    # Get precision-recall for current validation epoch:
                    prec_0 += (
                        y_pred[y_pred<0.5] == y_true[y_pred<0.5]
                    ).sum().item()
                    prec_1 += (
                        y_pred[y_pred>0.5] == y_true[y_pred>0.5]
                    ).sum().item()
                    reca_0 += (
                        y_pred[y_true<0.5] == y_true[y_true<0.5]
                    ).sum().item()
                    reca_1 += (
                        y_pred[y_true>0.5] == y_true[y_true>0.5]
                    ).sum().item()
                    prec_n_0 += torch.numel(y_pred[y_pred<0.5])
                    prec_n_1 += torch.numel(y_pred[y_pred>0.5])
                    reca_n_0 += torch.numel(y_true[y_true<0.5])
                    reca_n_1 += torch.numel(y_true[y_true>0.5])

                    # Check if any are zero, and if so, set to 1 sample, simply so doesn't crash:
                    # (Note: Just effects printing output)
                    if prec_n_0 == 0:
                        prec_n_0 = 1
                    if prec_n_1 == 0:
                        prec_n_1 = 1
                    if reca_n_0 == 0:
                        reca_n_0 = 1
                    if reca_n_1 == 0:
                        reca_n_1 = 1
            print("Precision (Class 0): {:4.3f}".format(prec_0/prec_n_0))
            print("Recall (Class 0): {:4.3f}".format(reca_0/reca_n_0))
            print("Precision (Class 1): {:4.3f}".format(prec_1/prec_n_1))
            print("Recall (Class 1): {:4.3f}".format(reca_1/reca_n_1))

            #y_pred_all = np.concatenate(y_pred_all)
            #y_true_all = np.concatenate(y_true_all)

            #from sklearn.metrics import classification_report
            #print(classification_report(y_true_all, y_pred_all))

            total_val_loss /= len(val_loader)
            total_val_acc = running_val_acc / running_sample_count
            print(
                "Validation loss = {:.4e}   acc = {:4.2f}%".format(
                    total_val_loss,
                    100*total_val_acc))

            # Save model:
            os.makedirs(self.model_path, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': total_val_loss,
            }, '%s/model_%03d_%f.pt' % (self.model_path, epoch, total_val_loss))

        print(
            "Training finished, took {:.2f}s".format(
                time.time() -
                training_start_time))

    def predict(self, data_loader):
        from torch.autograd import Variable
        import time

        for inputs, labels in val_loader:

            # Wrap tensors in Variables
            inputs, labels = Variable(
                inputs.to(device)), Variable(
                labels.to(device))

            # Forward pass only
            val_outputs = self.network(inputs)


def find_best_model(model_path="phaselink_model"):
    """Function to find best model.
    Note: Currently uses a very basic selection method."""
    # Plot model training and validation loss to select best model:

    # Write the models loss function values to file:
    models_fnames = list(glob.glob(os.path.join(model_path, "model_???_*.pt")))
    models_fnames.sort()
    val_losses = []
    f_out = open(os.path.join(model_path, 'val_losses.txt'), 'w')
    for model_fname in models_fnames:
        model_curr = torch.load(model_fname)
        val_losses.append(model_curr['loss'])
        f_out.write(' '.join((model_fname, str(model_curr['loss']), '\n')))
        del(model_curr)
        gc.collect()
    f_out.close()
    val_losses = np.array(val_losses)
    print("Written losses to file: ", os.path.join(model_path, 'val_losses.txt'))

    # And select approximate best model (approx corner of loss curve):
    approx_corner_idx = np.argwhere(val_losses < np.average(val_losses))[0][0]
    print("Model to use:", models_fnames[approx_corner_idx])

    # And plot:
    plt.figure()
    plt.plot(np.arange(len(val_losses)), val_losses)
    plt.hlines(val_losses[approx_corner_idx], 0, len(val_losses), color='r', ls="--")
    plt.ylabel("Val loss")
    plt.xlabel("Epoch")
    plt.show()

    # And convert model to use to universally usable format (GPU or CPU):
    model = StackedGRU().cuda(device)
    checkpoint = torch.load(models_fnames[approx_corner_idx], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    torch.save(model, os.path.join(model_path, 'model_to_use.gpu.pt'), _use_new_zipfile_serialization=False)
    new_device = "cpu"
    model = model.to(new_device)
    torch.save(model, os.path.join(model_path, 'model_to_use.cpu.pt'), _use_new_zipfile_serialization=False)
    del model
    gc.collect()

    print("Found best model and written out to", model_path, "for GPU and CPU.")


#----------------------------------------------- End: Define main functions -----------------------------------------------


#----------------------------------------------- Run script -----------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python phaselink_train.py config_json")
        print("E.g. python phaselink_train.py params.json")
        sys.exit()
    with open(sys.argv[1], "r") as f:
        params = json.load(f)

    # Get device (cpu vs gpu) specified:
    device = torch.device(params["device"])
    if params["device"][0:4] == "cuda":
        torch.cuda.empty_cache()
        enable_amp = True
    else:
        enable_amp = False
    if enable_amp:
        import apex.amp as amp

    # Get training info from param file:
    n_epochs = params["n_epochs"] #100

    # Load in training dataset:
    X = np.load(params["training_dset_X"])
    Y = np.load(params["training_dset_Y"])
    print("Training dataset info:")
    print("Shape of X:", X.shape, "Shape of Y", Y.shape)
    dataset = MyDataset(X, Y, device)

    # Get dataset info:
    n_samples = len(dataset)
    indices = list(range(n_samples))

    # Set size of training and validation subset:
    n_test = int(0.1*X.shape[0])
    validation_idx = np.random.choice(indices, size=n_test, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    # Specify samplers:
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    # Load training data:
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        shuffle=False,
        sampler=validation_sampler
    )

    stackedgru = StackedGRU()
    stackedgru = stackedgru.to(device)
    #stackedgru = torch.nn.DataParallel(stackedgru,
    #    device_ids=['cuda:2', 'cuda:3', 'cuda:4', 'cuda:5'])

    if enable_amp:
        #amp.register_float_function(torch, 'sigmoid')
        from apex.optimizers import FusedAdam
        optimizer = FusedAdam(stackedgru.parameters())
        stackedgru, optimizer = amp.initialize(
            stackedgru, optimizer, opt_level='O2')
    else:
        optimizer = torch.optim.Adam(stackedgru.parameters())

    model = Model(stackedgru, optimizer, model_path='./phaselink_model')
    print("Begin training process.")
    model.train(train_loader, val_loader, n_epochs, enable_amp=enable_amp)

    # And select and assign best model:
    find_best_model(model_path="phaselink_model")

    print("Finished.")
