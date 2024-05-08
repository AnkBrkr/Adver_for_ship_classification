import matplotlib.pyplot as plt
# %matplotlib inline

import os
import argparse
import random
import numpy as np
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import Accuracy, Precision, Recall, ConfusionMatrix, F1Score as F1
from torchsummary import summary

from advertorch.attacks import LinfPGDAttack
from advertorch.defenses import BinaryFilter
from advertorch.defenses import BitSqueezing

from nauta.model.builder import get_model
from nauta.dataset import get_split_dataloader
from nauta.tools.utils import plot_confusion_matrix, create_dir


config_path = '/home/psyop/Desktop/underwater_snd/config_files/vtuadfeature.yaml'
pth_path = '/home/psyop/Desktop/results/final_model/'
pth_name = 'best.pth'


def defense(poisoned:Tensor, cln:Tensor):
    """Apply the defense to the poisoned data and compare the results with the clean data.

    Args:
        poisoned (Tensor): The poisoned data tensor.
        cln (Tensor): The clean data tensor.

    Returns:
        Tensor: The defense applied to the poisoned data.
        Tensor: The defense applied to the clean data.
    """
    bits_queezing = BitSqueezing(bit_depth=5)
    binary_filter = BinaryFilter()
    defense = nn.Sequential(
        bits_queezing, 
        )
    
    adv = poisoned
    adv_defended = defense(adv)
    cln_defended = defense(cln)

    return adv_defended, cln_defended



def evaluate(model, dataloader, metric, eval_dir, condition, device='cpu'):
    """Perform an evaluation on the loaded model

    Args:
        model (nn.Module): The model to be used for evaluation.
        dataloader (Dataset): The dataloader object for the test dataset.
        metric (Dict): A dict containing the name and object of the metrics from torchmetrics.
        eval_dir (os.Path): The path where the artifacts will be saved
        device (str, optional): The device to load the tensors. Defaults to 'cpu'.
        condition: clean, untargeted, targeted
    """
    model.eval()
    data_info = []
    data_info.append(f"dataset_size,{len(dataloader.dataset)}")
    metrics = metric
    for input_data, target_data in tqdm(dataloader, position=0, leave=True):
        input_data = input_data.to(device)
        target_data = target_data.to(device)

        if condition == 'clean':
            prediction = model(input_data)
            for metric in metrics:
                metrics[metric].update(prediction, target_data)

        else:
            # Construct a LinfPGDAttack adversary instance
            adversary = LinfPGDAttack(
                model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.15,
                nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
                targeted=False)
            
            if condition == 'untargeted':    
                adv_untargeted = adversary.perturb(input_data, target_data)
                prediction = model(adv_untargeted)
                for metric in metrics:
                    metrics[metric].update(prediction, target_data)

            elif condition == 'targeted':
                target_specified = torch.ones_like(target_data) * 0
                adversary.targeted = True
                adv_targeted = adversary.perturb(input_data, target_specified)
                prediction = model(adv_targeted)
                for metric in metrics:
                    metrics[metric].update(prediction, target_data)

    for metric in metrics:
        value = metrics[metric].compute()
        if metric == "ConfusionMatrix":
            print(f"[Test_{metric}_{condition}]:\n{value}")
            cm_fig_norm = plot_confusion_matrix(
                value.cpu().detach().numpy(), class_names=dataloader.dataset.class_mapping.keys()
            )
            cm_fig_norm.savefig(os.path.join(eval_dir, f"confusion_{condition}.svg"))
            cm_fig = plot_confusion_matrix(
                value.cpu().detach().numpy(), class_names=dataloader.dataset.class_mapping.keys(), normalize=False
            )
            cm_fig.savefig(os.path.join(eval_dir, f"confusion_not_norm_{condition}.svg"))
        else:
            print(f"[Test {metric}]: {value}")
            data_info.append(f"{metric.lower()},{value}")
        metrics[metric].reset()

        with open(os.path.join(eval_dir, f"evaluation_{condition}.csv"), "w") as file:
            for line in data_info:
                file.write(f"{line}\n")


def def_n_evaluate(model, dataloader, metric, eval_dir, condition, device='cpu'):
    model.eval()
    data_info = []
    data_info.append(f"dataset_size,{len(dataloader.dataset)}")
    metrics = metric
    pth_defended = create_dir(os.path.join(eval_dir,"defended/"))

    for input_data, target_data in tqdm(dataloader, position=0, leave=True):
        input_data = input_data.to(device)
        target_data = target_data.to(device)

        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.15,
            nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
            targeted=False
        )
        adv_untargeted = adversary.perturb(input_data, target_data)

        adv_defended, cln_defended = defense(adv_untargeted, input_data)

        pred_clean = model(input_data)
        pred_adv = model(adv_untargeted)
        pred_adv_def = model(adv_defended)
        pred_cln_def = model(cln_defended)

        for metric in metrics:
            if condition == 'clean':
                metrics[metric].update(pred_clean, target_data)
            elif condition == 'untargeted':
                metrics[metric].update(pred_adv, target_data)
            elif condition == 'defended':
                metrics[metric].update(pred_adv_def, target_data)
            elif condition == 'cln_defended':
                metrics[metric].update(pred_cln_def, target_data)
        

    for metric in metrics:
        value = metrics[metric].compute()
        if metric == "ConfusionMatrix":
            print(f"[Test_{metric}_{condition}]:\n{value}")
            cm_fig_norm = plot_confusion_matrix(
                value.cpu().detach().numpy(), class_names=dataloader.dataset.class_mapping.keys()
            )
            cm_fig_norm.savefig(os.path.join(pth_defended, f"confusion_{condition}.svg"))
            cm_fig = plot_confusion_matrix(
                value.cpu().detach().numpy(), class_names=dataloader.dataset.class_mapping.keys(), normalize=False
            )
            cm_fig.savefig(os.path.join(pth_defended, f"confusion_not_norm_{condition}.svg"))
        else:
            print(f"[Test {metric}]: {value}")
            data_info.append(f"{metric.lower()},{value}")
        metrics[metric].reset()

        with open(os.path.join(pth_defended, f"evaluation_{condition}.csv"), "w") as file:
            for line in data_info:
                file.write(f"{line}\n")



if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load the configuration file
    with open(config_path, 'r') as file:
        args_list = yaml.load(file, Loader=yaml.FullLoader)

    # declare and initialize the model
    model = get_model(args_list, device=device)
    model_weights = torch.load(os.path.join(pth_path, pth_name))
    model.load_state_dict(model_weights)
    model.to(device)
    model.eval()

    # Initialize the metrics.
    num_of_classes = args_list["model"]["num_of_classes"]
    input_channels = args_list["model"]["input_channels"]
    eval_dir = create_dir(os.path.join(args_list["paths"]["output_dir"], "evaluation_adver"))
    print(f"\n\nStarting inference\n")
    print(f"Saving at {eval_dir}...")

    # get the test dataloader and initialize the metrics
    test_dataloader = get_split_dataloader(args_list, split="test")

    accuracy = Accuracy(average='macro',task='multiclass', num_classes=num_of_classes).to(device)
    precision = Precision(average='macro',task='multiclass' , num_classes=num_of_classes).to(device)
    recall = Recall(average='macro',task='multiclass' , num_classes=num_of_classes).to(device)
    f1 = F1(average='macro',task='multiclass' , num_classes=num_of_classes).to(device)
    confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=num_of_classes).to(device)

    metrics = {
        "Accuracy":accuracy,
        "Precision":precision,
        "Recall":recall,
        "F1":f1,
        "ConfusionMatrix":confusion_matrix,
    }

    evaluate(model, test_dataloader, metrics, eval_dir,
             condition='clean', device=device)
    evaluate(model, test_dataloader, metrics, eval_dir,
             condition='untargeted', device=device)
    evaluate(model, test_dataloader, metrics, eval_dir,
             condition='targeted', device=device)
    
    # def_n_evaluate(model, test_dataloader, metrics, eval_dir,
    #             condition='defended', device=device)
    # def_n_evaluate(model, test_dataloader, metrics, eval_dir,
    #             condition='cln_defended', device=device)
    
    