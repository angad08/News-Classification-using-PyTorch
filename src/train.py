from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import tqdm.notebook as tq
import wandb
import torchmetrics
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification,AutoModelForTokenClassification, AutoTokenizer
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from pathlib import Path

# Determine which device on import, and then use that elsewhere.
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)


def plot_confusion_matrix(cm, class_names):
    '''
        cm: the confusion matrix that we wish to plot
        class_names: the names of the classes 
    '''

    # this normalizes the confusion matrix
    cm = cm.astype(np.float32) / cm.sum(axis=1)[:, None]
    
    df_cm = pd.DataFrame(cm, class_names, class_names)
    ax = sn.heatmap(df_cm, annot=True, cmap='flare')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.show()
    
def count_classes(preds):
    '''
    Counts the number of predictions per class given preds, a tensor
    shaped [batch, n_classes], where the maximum per preds[i]
    is considered the "predicted class" for batch element i.
    '''
    pred_classes = preds.argmax(dim=1)
    n_classes = preds.shape[1]
    return [(pred_classes == c).sum().item() for c in range(n_classes)]

def train_epoch(epoch, model, optimizer, criterion, loader, num_classes, device):
    '''
    Train the model on the entire training set precisely once (one epoch).
    Lab 6 has a very similar function.
    '''

    model.train()

    # Initialize metrics
    # TODO: Task 1b - initialize the following torchmetrics metrics
    #       the average epoch loss per example
    #       accuracy
    #       unweighted average recall
    epoch_trainmean= torchmetrics.MeanMetric().to(device)
    epoch_trainaccuracy= torchmetrics.Accuracy(num_classes=num_classes,task="multiclass").to(device)
    epoch_trainuar= torchmetrics.Recall(num_classes=num_classes,average="macro",task="multiclass").to(device)

    #putting the model to the device
    model.to(device)

    

  
    for i,(inputs, lbls) in enumerate(loader):
        inputs, lbls = inputs.to(device), lbls.to(device)
        # Update model weights
        # TODO: Task 1b - Use the batch to update the weights of the model
        #zero the parameter gradients of the model
        optimizer.zero_grad()
        #forward pass to get the output
        outputs = model(inputs)
        #calculate the loss
        loss = criterion(outputs, lbls)
        #backpropagate the loss
        loss.backward()
        #update the parameters
        optimizer.step()

        # Accumulate metrics
        # TODO: Task 1b - accumulate each of the 3 metrics you initialized.
        epoch_trainmean.update(loss)
        epoch_trainaccuracy.update(outputs,lbls)
        epoch_trainuar.update(outputs,lbls)
        



    # Calculate epoch metrics, and store in a dictionary for wandb
    # TODO Task 1b - compute the three metrics
    epoch_trainmean=epoch_trainmean.compute()
    epoch_trainaccuracy=epoch_trainaccuracy.compute()
    epoch_trainuar=epoch_trainuar.compute()
    metrics_dict = {
        'Loss_train':epoch_trainmean,
        'Accuracy_train':epoch_trainaccuracy,
        'UAR_train':epoch_trainuar
    }

    return metrics_dict

def val_epoch(epoch, model, criterion, loader, num_classes, device):
    '''
    Evaluate the model on the entire validation set.
    '''
    model.eval()
    
    # Initialize metrics
    # TODO: Task 1b - initialize the following torchmetrics metrics
    #       the average epoch loss per example
    #       accuracy
    #       unweighted average recall
    epoch_valmean= torchmetrics.MeanMetric().to(device)
    epoch_valaccuracy= torchmetrics.Accuracy(num_classes=num_classes,task="multiclass").to(device)
    epoch_valuar= torchmetrics.Recall(num_classes=num_classes,average="macro",task="multiclass").to(device)

    
    # TODO: Task 1c - initialize a confusion matrix torchmetrics object
    metric = torchmetrics.ConfusionMatrix(num_classes=num_classes,task="multiclass").to(device)




    predictions = []  # Empty list to store the predictions
    for inputs, lbls in loader:
        inputs, lbls = inputs.to(device), lbls.to(device)

        # TODO Task 1b - Obtain validation loss (use torch.no_grad())
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, lbls)
        # Store the predictions
        predictions.append(torch.argmax(outputs,1))

            

        
        # Accumulate metrics
        # TODO: Task 1b - accumulate each of the 3 metrics you initialized
        #       Look at trainer.py of lab 6 for inspiration.
        #       This will take the loss, outputs and lbls from this batch
        #       to update each of the metric object's internal state.
         
        epoch_valmean.update(loss)
        epoch_valaccuracy.update(outputs,lbls)
        epoch_valuar.update(outputs,lbls)

    
        # TODO: Task 1c - acculmate confusion matrix
        metric.update(outputs, lbls)

        

 
    # Print the predictions
    #print("Predictions:", torch.argmax(outputs,1))
         
    # Calculate epoch metrics, and store in a dictionary for wandb
    # TODO Task 1b - compute the three metrics 
    epoch_valmean=epoch_valmean.compute()
    epoch_valaccuracy=epoch_valaccuracy.compute()
    epoch_valuar=epoch_valuar.compute()
    metrics_dict = {
        'Loss_val':epoch_valmean,
        'Accuracy_val':epoch_valaccuracy,
        'UAR_val':epoch_valuar
    }


    # Compute the confusion matrix
    # TODO: Task 1c - compute the confusion matrix and store it in cm
    cm = metric.compute().detach().cpu().numpy() 
    return metrics_dict,predictions,cm
    


def train_model(model, train_loader, val_loader, optimizer, criterion,
                class_names, n_epochs):
                
    num_classes = len(class_names)
    model.to(device)
    # # Initialise Weights and Biases (wandb) project
    # if ident_str is None:
    #   ident_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # exp_name = f"{model.__class__.__name__}_{ident_str}"
    # run = wandb.init(project=project_name, name=exp_name)

    try:
        # Train by iterating over epochs
        for epoch in tq.tqdm(range(n_epochs), total=n_epochs, desc='Epochs'):
            train_metrics_dict = train_epoch(epoch, model, optimizer, criterion,
                    train_loader, num_classes, device)
                    
            val_metrics_dict,predictions,cm = val_epoch(epoch, model, criterion, 
                    val_loader, num_classes, device)
            # wandb.log({**train_metrics_dict, **val_metrics_dict})
        print(val_metrics_dict)
    finally:
        print("Something")

    # Plot confusion matrix from results of last val epoch
    # TODO Task 1c - call plot_confusion_matrix with appropriate arguments.
    plot_confusion_matrix(cm, class_names)
    # Save the model weights to "saved_models/"
    # TODO Task 2b - Save model weights
    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pth")
