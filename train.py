import time

import torch
import torch.nn as nn
import torchvision
import torch

torch.cuda.empty_cache()
from network import TransformNetwork
from images_unit import get_transformer, imsave, imload, ImageFolder

mse_criterion = torch.nn.MSELoss(reduction='mean')


def extract_features(model, input, layers) -> list:
    features = list()
    for index, layer in enumerate(model):
        input = layer(input)
        if index in layers:
            features.append(input)
    return features


def calc_Content_loss(features, targets):
    content_loss = 0
    for f, t in zip(features, targets):
        content_loss += mse_criterion(f, t)
    return content_loss * (1 / len(features))


def gram(in_put):
    b, c, h, w = in_put.size()
    g = torch.bmm(in_put.view(b, c, h * w), in_put.view(b, c, h * w).transpose(1, 2))
    return g.div(h * w)


def calc_Style_loss(features, targets):
    style_loss = 0
    for f, t in zip(features, targets):
        style_loss += mse_criterion(gram(f), gram(t))
    return style_loss * 1 / len(features)


def calc_TV_loss(features):
    tv_loss = torch.mean(abs(features[:, :, :, :-1] - features[:, :, :, 1:]))
    tv_loss += torch.mean(abs(features[:, :, :-1, :] - features[:, :, 1:, :]))
    return tv_loss


def network_train(args):
    # choose device to train
    device = torch.device('cuda' if args.cuda_device_no >= 0 else 'cpu')

    # init transformnetwork
    transform_network = TransformNetwork()
    transform_network = transform_network.to(device)

    train_dataset = ImageFolder(args.train_content, get_transformer(args.imsize, args.cropsize))

    # loss network
    loss_network = torchvision.models.__dict__[args.vgg_flag](pretrained=True).features.to(device)

    # optimizer
    optimizer = torch.optim.Adam(transform_network.parameters(), lr=args.lr)

    # Target style image load
    target_style_image = imload(args.train_style, imsize=args.imsize).to(device)
    b, c, h, w = target_style_image.size()
    target_style_image = target_style_image.expand(args.batchs, c, h, w)

    # loss logs
    loss_logs = {'content_loss': [], 'style_loss': [], 'tv_loss': [], 'total_loss': []}

    for iteration in range(args.max_iter):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchs, shuffle=True)
        image = next(iter(train_dataloader))
        image = image.to(device)

        output_image = transform_network(image)

        target_style_features = extract_features(loss_network, target_style_image, args.style_layers)
        target_content_features = extract_features(loss_network, image, args.content_layers)

        output_style_features = extract_features(loss_network, output_image, args.style_layers)
        output_content_features = extract_features(loss_network, output_image, args.content_layers)

        content_loss = calc_Content_loss(output_content_features, target_content_features)
        style_loss = calc_Style_loss(output_style_features, target_style_features)
        tv_loss = calc_TV_loss(output_image)
        total_loss = content_loss * args.content_weight + style_loss * args.style_weight + tv_loss * args.tv_weight

        loss_logs['content_loss'].append(content_loss.item())
        loss_logs['style_loss'].append(style_loss.item())
        loss_logs['tv_loss'].append(tv_loss.item())
        loss_logs['total_loss'].append(total_loss.item())
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if iteration % args.check_iter == 0:
            str_ = '%s: iteration: [%d/%d/],\t' % (time.ctime(), iteration, args.max_iter)
            for key, value in loss_logs.items():
                str_ += '%s: %2.2f,\t' % (key, sum(value[-100:]) / 100)
            print(str_)
            imsave(output_image.cpu(), args.save_path + "pngpng_train.png")
            torch.save(transform_network.state_dict(), args.save_path + "transform_network.pth")

    torch.save(loss_logs, args.save_path + "loss_logs.pth")
    torch.save(transform_network.state_dict(), args.save_path + "transform_network.pth")
    return transform_network

