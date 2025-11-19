import argparse
import datetime
import os
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models import spiking_resnet20
from spikingjelly.clock_driven.neuron import LIFNode
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven import surrogate as surrogate_sj
from utils import Bar, AverageMeter, accuracy
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchtoolbox.transform import Cutout
import collections
import random
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from typing import Callable
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = 'cuda:4'
seed = 2025
data_dir = '../datasets'
out_dir = './logs_cifar10_res20_zo_subspace_rge_q10'
ckpt_dir = './logs_cifar10_res20'
num_classes = 10

# models
model = 'spiking_res20'
t_step = 4
tau = 1.1
resume = None
pre_train = os.path.join(ckpt_dir, 'checkpoint_0.pth')

# loss function
criterion_mse = nn.MSELoss()
loss_lambda = 0.05
loss_means = 1.

# optimizer
lr = 1
momentum = 0.9
weight_decay = 0.0

# training
epochs = 40
bs = 128

#subspace
window_size = 150
pca_components = 60

# zo
epsilon = 1.0
query = 10

_seed_ = seed
random.seed(_seed_)
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
np.random.seed(_seed_)


class BPTTNeuron(LIFNode):
    def __init__(self, tau: float = 1.1, decay_input: bool = False, v_threshold: float = 1.,
            v_reset: float = None, surrogate_function: Callable = surrogate_sj.PiecewiseQuadratic(),
            detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)


def inference_process(net, frame, label, backward=False):
    total_fr = None
    total_loss = 0.0
    for t in range(t_step):
        input_frame = frame
        if t == 0:
            out_fr = net(input_frame)
            total_fr = out_fr.clone().detach()
        else:
            out_fr = net(input_frame)
            total_fr += out_fr.clone().detach()
        label_one_hot = torch.zeros_like(out_fr).fill_(loss_means).to(out_fr.device)
        mse_loss = criterion_mse(out_fr, label_one_hot)
        loss = ((1 - loss_lambda) * F.cross_entropy(out_fr, label) + loss_lambda * mse_loss) / t_step
        total_loss += loss.item()

        if backward:
            loss.backward(retain_graph=True)

    functional.reset_net(net)
    return total_fr, total_loss


def test(net, test_data_loader):
    net.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    test_loss = 0
    test_acc = 0
    test_samples = 0
    batch_idx = 0
    with torch.no_grad():
        for frame, label in test_data_loader:
            batch_idx += 1
            frame = frame.float().to(device)
            label = label.to(device)

            total_fr, total_loss = inference_process(net, frame, label, backward = False)

            test_samples += label.numel()
            test_loss += total_loss * label.numel()
            test_acc += (total_fr.argmax(1) == label).float().sum().item()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(total_fr.data, label.data, topk=(1, 5))
            losses.update(total_loss, frame.size(0))
            top1.update(prec1.item(), frame.size(0))
            top5.update(prec5.item(), frame.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    test_loss /= test_samples
    test_acc /= test_samples

    return test_acc, test_loss


def train_zo_subspace(net, train_data_loader, optimizer, proj_matrix):

    pos_pertubation_net = spiking_resnet20.__dict__[model](neuron=BPTTNeuron, num_classes=num_classes)
    pos_pertubation_net.to(device)
    neg_pertubation_net = spiking_resnet20.__dict__[model](neuron=BPTTNeuron, num_classes=num_classes)
    neg_pertubation_net.to(device)

    net.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_data_loader))

    train_loss = 0
    train_acc = 0
    train_samples = 0
    batch_idx = 0


    for frame, label in train_data_loader:
        batch_idx += 1

        label = label.to(device)
        frame = frame.float().to(device)

        optimizer.zero_grad()

        grad_estimate = torch.zeros(proj_matrix.shape[1]).to(device)
        param_vec = net.get_flatten_param()
        # dim = proj_matrix.shape[0]

        for q in range(query):
            pos_pertubation_net.load_state_dict(net.state_dict())
            neg_pertubation_net.load_state_dict(net.state_dict())

            # perturb = proj_matrix[d, :]
            perturb_sub = torch.randn(proj_matrix.shape[0]).to(device)
            perturb = torch.matmul(proj_matrix.transpose(0, 1), perturb_sub)

            param_vec_perturb = param_vec + epsilon * perturb
            pos_pertubation_net.set_param_from_flatten_param(param_vec_perturb.detach())

            # forward inference start
            with torch.no_grad():
                _, batch_loss_pos = inference_process(pos_pertubation_net, frame, label, backward = False)
            # forward inference end

            param_vec_perturb = param_vec - epsilon * perturb
            neg_pertubation_net.set_param_from_flatten_param(param_vec_perturb.detach())

            # backward inference start
            with torch.no_grad():
                _, batch_loss_neg = inference_process(neg_pertubation_net, frame, label, backward = False)
            # backward inference end

            grad_estimate += perturb * (batch_loss_pos - batch_loss_neg) / (2.0*epsilon) / query

        # update grad
        optimizer.zero_grad()
        net.set_grad_from_flatten_vec(grad_estimate.detach())
        
        optimizer.step()

        with torch.no_grad():
            total_fr, batch_loss = inference_process(net, frame, label, backward = False)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(total_fr.data, label.data, topk=(1, 5))
        losses.update(batch_loss, frame.size(0))
        top1.update(prec1.item(), frame.size(0))
        top5.update(prec5.item(), frame.size(0))

        train_samples += label.numel()
        train_acc += (total_fr.argmax(1) == label).float().sum().item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx,
                    size=len(train_data_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()

    bar.finish()

    train_loss /= train_samples
    train_acc /= train_samples

    return train_acc, train_loss


def main():

    ########################################################
    # data preparing
    ########################################################
    dataloader = datasets.CIFAR10
    num_classes = 10
    normalization_mean = (0.4914, 0.4822, 0.4465)
    normalization_std = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        Cutout(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_std),
    ])

    trainset = dataloader(root=data_dir, train=True, download=True, transform=transform_train)
    train_data_loader = data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

    testset = dataloader(root=data_dir, train=False, download=False, transform=transform_test)
    test_data_loader = data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=8)


    ##########################################################
    # model preparing
    ##########################################################
    net = spiking_resnet20.__dict__[model](neuron=BPTTNeuron, num_classes=num_classes)
    print('using resnet model.')
    print('Total Parameters: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    net.to(device)

    ##########################################################
    # optimizer preparing
    ##########################################################
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ##########################################################
    # loading models from checkpoint
    ##########################################################
    start_epoch = 0
    max_test_acc = 0

    if resume:
        print('resuming...')
        checkpoint = torch.load(resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
        print('start epoch:', start_epoch, ', max test acc:', max_test_acc)

    if pre_train:
        checkpoint = torch.load(pre_train, map_location='cpu')
        state_dict2 = collections.OrderedDict([(k, v) for k, v in checkpoint['net'].items()])
        net.load_state_dict(state_dict2)
        print('use pre-trained model, max test acc:', checkpoint['max_test_acc'])

    ##########################################################
    # output setting
    ##########################################################

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')
    else:
        print('out dir already exists:', out_dir)

    # save the initialization of parameters
    checkpoint = {
        'net': net.state_dict(),
        'epoch': 0,
        'max_test_acc': 0.0
    }
    torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_0.pth'))

    writer = SummaryWriter(os.path.join(out_dir, 'logs'), purge_step=start_epoch)

    ##########################################################
    # constructing subspace
    ##########################################################
    weight_vec_list = []
    for i in tqdm(range(0, window_size)):
        tmp_model = spiking_resnet20.__dict__[model](neuron=BPTTNeuron, num_classes=num_classes)
        checkpoint = torch.load(os.path.join(ckpt_dir, f'checkpoint_{i}.pth'))
        state_dict = checkpoint['net']
        tmp_model.load_state_dict(state_dict)
        weight_vec_list.append(tmp_model.get_flatten_param().unsqueeze(0))
    weight_matrix = torch.concat(weight_vec_list, dim=0)
    print("weight_matrix is: ", weight_matrix.shape)

    pca = PCA(n_components=pca_components)
    pca.fit_transform(weight_matrix.detach().cpu().numpy())
    proj_matrix = torch.from_numpy(pca.components_)
    print("eigen is " + str(pca.explained_variance_))
    print("ratio is " + str(pca.explained_variance_ratio_))
    proj_matrix = proj_matrix.to(device)

    ##########################################################
    # training and testing
    ##########################################################

    for epoch in range(start_epoch, epochs):
        ############### training ###############
        start_time = time.time()

        train_acc, train_loss = train_zo_subspace(net, train_data_loader, optimizer, proj_matrix)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        lr_scheduler.step()

        ############### testing ###############

        test_acc, test_loss = test(net, test_data_loader)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        ############### saving checkpoint ###############
        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))
        # torch.save(checkpoint, os.path.join(out_dir, f'checkpoint_{epoch+1}.pth'))

        total_time = time.time() - start_time
        print(f'epoch={epoch}:')
        print(f'train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, max_test_acc={max_test_acc:.4f}, total_time={total_time:.4f}, escape_time={(datetime.datetime.now()+datetime.timedelta(seconds=total_time * (epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}')
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")

if __name__ == '__main__':
    main()
