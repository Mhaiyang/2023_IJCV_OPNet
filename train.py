"""
 @Time    : 2021/9/2 20:52
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : 2023_IJCV_OPNet
 @File    : train.py
 @Function:
 
"""
import datetime
import time
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

import joint_transforms
from config import cod_training_root
from config import backbone_path
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from OPNet import OPNet

import loss

cudnn.benchmark = True

torch.manual_seed(2021)
device_ids = [1]

ckpt_path = './ckpt'
exp_name = 'OPNet'

args = {
    'epoch_num': 200,
    'train_batch_size': 14,
    'last_epoch': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 416,
    'save_point': [],
    'poly_train': True,
    'optimizer': 'SGD',
}

print(torch.__version__)

# Path.
check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
vis_path = os.path.join(ckpt_path, exp_name, 'log')
check_mkdir(vis_path)
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')
writer = SummaryWriter(log_dir=vis_path, comment=exp_name)

# Transform Data.
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args['scale'], args['scale']))
])
img_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

# Prepare Data Set.
train_set = ImageFolder(cod_training_root, joint_transform, img_transform, target_transform)
print("Train set: {}".format(train_set.__len__()))
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True)

total_epoch = args['epoch_num'] * len(train_loader)

# loss function
structure_loss = loss.structure_loss().cuda(device_ids[0])

def main():
    print(args)
    print(exp_name)

    net = OPNet(backbone_path).cuda(device_ids[0]).train()

    if args['optimizer'] == 'Adam':
        print("Adam")
        optimizer = optim.Adam([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ])
    else:
        print("SGD")
        optimizer = optim.SGD([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('Training Resumes From \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        total_epoch = (args['epoch_num'] - int(args['snapshot'])) * len(train_loader)
        print(total_epoch)

    net = nn.DataParallel(net, device_ids=device_ids)
    print("Using {} GPU(s) to Train.".format(len(device_ids)))

    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)
    writer.close()

def train(net, optimizer):
    curr_iter = 1
    start_time = time.time()

    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):
        loss_record, loss_4_c_record, loss_3_c_record, loss_2_c_record, \
        loss_4_t_record, loss_3_t_record, loss_2_t_record, loss_final_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), \
                                                                               AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        train_iterator = tqdm(train_loader, total=len(train_loader))
        for data in train_iterator:
            if args['poly_train']:
                base_lr = args['lr'] * (1 - float(curr_iter) / float(total_epoch)) ** args['lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda(device_ids[0])
            labels = Variable(labels).cuda(device_ids[0])

            optimizer.zero_grad()

            predict_4_c, predict_3_c, predict_2_c, predict_4_t, predict_3_t, predict_2_t, predict_final = net(inputs)

            loss_4_c = structure_loss(predict_4_c, labels)
            loss_3_c = structure_loss(predict_3_c, labels)
            loss_2_c = structure_loss(predict_2_c, labels)
            loss_4_t = structure_loss(predict_4_t, labels)
            loss_3_t = structure_loss(predict_3_t, labels)
            loss_2_t = structure_loss(predict_2_t, labels)
            loss_final = structure_loss(predict_final, labels)

            loss = loss_4_c + loss_3_c + loss_2_c + loss_4_t + loss_3_t + loss_2_t + 2 * loss_final

            loss.backward()

            optimizer.step()

            loss_record.update(loss.data, batch_size)
            loss_4_c_record.update(loss_4_c.data, batch_size)
            loss_3_c_record.update(loss_3_c.data, batch_size)
            loss_2_c_record.update(loss_2_c.data, batch_size)
            loss_4_t_record.update(loss_4_t.data, batch_size)
            loss_3_t_record.update(loss_3_t.data, batch_size)
            loss_2_t_record.update(loss_2_t.data, batch_size)
            loss_final_record.update(loss_final.data, batch_size)

            if curr_iter % 10 == 0:
                writer.add_scalar('loss', loss, curr_iter)
                writer.add_scalar('loss_4_c', loss_4_c, curr_iter)
                writer.add_scalar('loss_3_c', loss_3_c, curr_iter)
                writer.add_scalar('loss_2_c', loss_2_c, curr_iter)
                writer.add_scalar('loss_4_t', loss_4_t, curr_iter)
                writer.add_scalar('loss_3_t', loss_3_t, curr_iter)
                writer.add_scalar('loss_2_t', loss_2_t, curr_iter)
                writer.add_scalar('loss_final', loss_final, curr_iter)

            log = '[%3d], [%6d], [%.6f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f]' % \
                  (epoch, curr_iter, base_lr, loss_record.avg, loss_4_c_record.avg, loss_3_c_record.avg, loss_2_c_record.avg,
                   loss_4_t_record.avg, loss_3_t_record.avg, loss_2_t_record.avg, loss_final_record.avg)
            train_iterator.set_description(log)
            open(log_path, 'a').write(log + '\n')

            curr_iter += 1

        if epoch in args['save_point']:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            net.cuda(device_ids[0])

        if epoch >= args['epoch_num']:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
            print(exp_name)
            print("Optimization Have Done!")
            return

if __name__ == '__main__':
    main()
