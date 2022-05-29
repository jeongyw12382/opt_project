import wandb
import torch
import argparse
import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torch.optim import SGD, Adam, RAdam, RMSprop
from torchvision.models import resnet18, resnet34, vgg11, vgg13

from scheduler import *

dropout_dict = {0: 0, 1: 0.3, 2:0.6, 3: 0.9}

def transform(args):
    strength = 0.5 * args.noise_level / 3
    jitter = tf.ColorJitter(strength, strength, strength, strength)
    return tf.Compose([jitter, tf.ToTensor()])

def run(args):

    if args.dset == "mnist":
        train_dataset = MNIST("data", True, transform(args), download=True)
        test_dataset = MNIST("data", False, tf.ToTensor(), download=True)
    elif args.dset == "cifar10":
        train_dataset = CIFAR10("data", True, transform(args), download=True)
        test_dataset = CIFAR10("data", False, tf.ToTensor(), download=True)
    else:
        raise NameError(f"No dset named {args.dset}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.bsz, shuffle=True, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.bsz, shuffle=False, num_workers=4
    )

    dropout_ratio = dropout_dict[args.dropout_level]

    if args.model == "resnet":
        model = resnet18(pretrained=args.pretrained)
        model.fc = nn.Linear(512, 10, bias=True)
        if dropout_ratio > 0:
            model.layer4[1].conv2 = nn.Dropout(
                dropout_ratio, 
                model.layer4[1].conv2
            )
    elif args.model == "vggnet":
        model = vgg11(pretrained=args.pretrained)
        model.classifier[-1] = nn.Linear(4096, 10, bias=True)

    model = model.cuda()

    if args.optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr_init)
    elif args.optimizer == "sgd_momentum":
        optimizer = SGD(model.parameters(), lr=args.lr_init, momentum=0.9)
    elif args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr_init)
    elif args.optimizer == "rmsprop":
        optimizer = RMSprop(model.parameters(), lr=args.lr_init)
    elif args.optimizer == "radam":
        optimizer = RAdam(model.parameters(), lr=args.lr_init)
    else:
        raise NameError(f"No optimizer named {args.optimizer}")


    if args.lr_scheduler == "none":
        lr_scheduler_fn = ConstLR
    elif args.lr_scheduler == "linear_warmup_cosine_annealing":
        lr_scheduler_fn = LinearWarmupCosineAnnealing
    elif args.lr_scheduler == "cosine_annealing":
        lr_scheduler_fn = CosineAnnealing
    elif args.lr_scheduler == "exponential_decay":
        lr_scheduler_fn = ExponentialDecay

    lr_scheduler = lr_scheduler_fn(args.lr_init, args.global_step)
    
    wandb.init(project="opt_project")

    step = 0
    best_acc = -1000
    while step != args.global_step:

        model.train()
        for batch in tqdm.tqdm(train_dataloader):
            if step == args.global_step:
                break
            optimizer.zero_grad()
            curr_lr = lr_scheduler.get_lr(step)
            image, label = batch
            image = image.cuda()
            label = label.cuda()
            if image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)
            one_hot = torch.zeros(label.shape[0], 10, device="cuda")
            bsz = label.shape[0]
            idx = torch.arange(bsz, device="cuda") * 10
            one_hot.view(-1)[idx + label] = 0.8
            one_hot[one_hot == 0] = 0.2 / 9

            logit = F.softmax(model(image), dim=1)
            loss = -(one_hot * torch.log(logit)).sum(dim=1).mean()
            loss.backward()

            if step % 100 == 0:
                tqdm.tqdm.write(f"Loss ({step} / {args.global_step} : {loss.item()}")
                wandb.log({"train_loss": loss.item()})

            for g in optimizer.param_groups:
                g['lr'] = curr_lr

            optimizer.step()
            step += 1
        
        model.eval()
        
        preds, labels = [], []
        for batch in tqdm.tqdm(test_dataloader):
            curr_lr = lr_scheduler.get_lr(step)
            image, label = batch
            image = image.cuda()
            label = label.cuda()
            if image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)
            pred = model(image).argmax(dim=1)
            preds.append(pred)
            labels.append(label)
        
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        acc = (preds == labels).float().mean() * 100
        wandb.log({"test_acc": acc.item()})
        
        if best_acc < acc: 
            best_acc = acc
            wandb.log({"test_best_acc": best_acc.item()})


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dset", 
        type=str,
        choices=["mnist", "cifar10"],
        default="mnist"
    )

    parser.add_argument(
        "--optimizer", 
        type=str, 
        choices=["sgd", "sgd_momentum", "adam", "rmsprop", "radam"],
        default="sgd"
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=[
            "none", 
            "linear_warmup_cosine_annealing", 
            "cosine_annealing", 
            "exponential_decay"
        ],
        default="none"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet", "vggnet", "vit"],
        default="resnet"
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--bsz",
        type=int, 
        default=128,
    )

    parser.add_argument(
        "--lr_init",
        type=float,
        choices=[1.0, 0.1, 0.01, 0.001, 0.0001],
        default=0.1
    )

    parser.add_argument(
        "--global_step",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--dropout_level", 
        type=int,
        default=0
    )

    parser.add_argument(
        "--noise_level",
        type=int, 
        default=0
    )

    args = parser.parse_args()

    run(args)