import os

import torch
import torchvision
import numpy as np
from tensorboardX import SummaryWriter
global writer
from torchvision import transforms, datasets


total_step = 0


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def save(model, ckpt_num, dir_name):
    os.makedirs(dir_name, exist_ok=True)
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), os.path.join(dir_name, 'model_%s' % ckpt_num))
    else:
        torch.save(model.state_dict(), os.path.join(dir_name, 'model_%s' % ckpt_num))
    print('model saved!')


def fit(train_loader, model, loss_fn, optimizer, scheduler, nb_epoch,
        device, log_interval, start_epoch=0, save_model_to='/tmp/save_model_to'):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    global writer
    writer = SummaryWriter('runs')

    # Save pre-trained model
    save(model, 0, save_model_to)

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, nb_epoch):
        #scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(epoch, train_loader, model, loss_fn, optimizer, device, log_interval)

        log_dict = {'epoch': epoch + 1,
                    'epoch_total': nb_epoch,
                    'loss': float(train_loss),
                    }

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, nb_epoch, train_loss)

        for metric in metrics:
            log_dict[metric.name()] = metric.value()
            message += '\t{}: {}'.format(metric.name(), metric.value())


        print(message)
        print(log_dict)
        #if (epoch + 1) % 5 == 0:
        scheduler.step()
        save(model, epoch + 1, save_model_to)


def train_epoch(epoch, train_loader, model, loss_fn, optimizer, device, log_interval):

    for metric in loss_fn.metrics:
        metric.reset()

    model.train()
    global writer
    global total_step
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)

        data = tuple(d.to(device) for d in data)
        if target is not None:
            target = target.to(device)

        optimizer.zero_grad()
        if loss_fn.cross_entropy_flag:
            output_embedding, output_cross_entropy = model(*data)
            blended_loss, losses = loss_fn.calculate_loss(target, output_embedding, output_cross_entropy)
        else:
            output_embedding = model(*data)
            blended_loss, losses = loss_fn.calculate_loss(target, output_embedding)
        total_loss += blended_loss.item()
        blended_loss.backward()

        optimizer.step()

        # Print log
        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]'.format(
                batch_idx * len(data[0]), len(train_loader.dataset), 100. * batch_idx / len(train_loader))
            for name, value in losses.items():
                message += '\t{}: {:.6f}'.format(name, np.mean(value))
                #print("------")
                #print(total_step)
                #print(np.mean(value))
                #print(value)
                writer.add_scalar('data/loss', np.mean(value), total_step)

                #print(data[0].min())
                #print(data[0].max())

                image = data[0]
                """
                inv_normalize = transforms.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                    std=[1/0.229, 1/0.224, 1/0.255]
                )
                imgs = inv_normalize(data[0])
                """
                """
                unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                imgs = unnorm(data[0])*255

                print(imgs)
                grid = torchvision.utils.make_grid(imgs)
                writer.add_image('images', grid, 0)
                """

                image = (image - image.min()) / (image.max() - image.min())
                grid = torchvision.utils.make_grid(image)
                writer.add_image('images', grid, total_step)


                total_step += 1



            for metric in loss_fn.metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())



            print(message)


    total_loss /= (batch_idx + 1)
    return total_loss, loss_fn.metrics
