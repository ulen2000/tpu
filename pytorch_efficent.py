# coding=UTF-8
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
import json
from efficientnet_pytorch.model import EfficientNet
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
plt.switch_backend('agg')

use_gpu = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    #设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'
data_dir = './image_large'
batch_size = 64		#批次大小（通过loaddata函数打包）
lr = 0.01  		  #学习率
momentum = 0.9     #动量
num_epochs = 20     #训练轮次
input_size = 350    #数据集图像处理大小
class_num = 5     #分多少个类别
net_name = 'efficientnet-b8'	#需要用到的EfficientNet预训练模型名称
i=0
Species_id = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loaddata(data_dir, batch_size, set_name, shuffle):
    #对数据进行数据增强
    data_transforms = {
        'train': transforms.Compose([
            transforms.CenterCrop(input_size),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [set_name]}
    # num_workers=0 if CPU else =1
    #创建了一个 batch，生成真正网络的输入
    dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                      batch_size=batch_size,
                                                      shuffle=shuffle, num_workers=24) for x in [set_name]} ########num_workers=1
    data_set_sizes = len(image_datasets[set_name])
    return dataset_loaders, data_set_sizes


def train_model(model_ft, criterion, optimizer, lr_scheduler, num_epochs=50):
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    train_loss = []
    loss_all = []
    acc_all = []
    since = time.time()
    best_model_wts = model_ft.state_dict()
    best_acc = 0.0
    model_ft.train(True)
    with open("./txt/acc.txt", "w") as f:
        with open("./txt/log.txt", "w")as f2:
            for epoch in range(num_epochs):
                dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=batch_size, set_name='train',
                                                    shuffle=True)
                # print(dset_loaders)
                print('Data Size', dset_sizes)
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
                optimizer = lr_scheduler(optimizer, epoch)

                running_loss = 0.0
                running_corrects = 0
                count = 0

                for i, data in enumerate(dset_loaders['train'], 0):
                    length = len(dset_loaders['train'])
                    # print(data)
                    inputs, target = data
                    inputs, target = inputs.cuda(), target.cuda()

                    # if use_gpu:
                    #    inputs, target = Variable(inputs.cuda()), Variable(target.cuda())
                    # else:
                    #    inputs, target = Variable(inputs), Variable(target)

                    # 训练
                    optimizer.zero_grad()  # 加
                    # forward + backward
                    with torch.no_grad():  # 无
                        outputs = model_ft(inputs)  # model_ft=net
                    loss = criterion(outputs, target)
                    loss = loss.requires_grad_()
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += predicted.eq(target.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                             100. * float(correct) / float(total)))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                                100. * float(correct) / float(total)))
                    f2.write('\n')
                    f2.flush()

                    count += 1
                    if count % 30 == 0 or outputs.size()[0] < batch_size:
                        print('Epoch:{}: loss:{:.3f}'.format(epoch, loss.item()))
                        train_loss.append(loss.item())

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(predicted == target.data)

                epoch_loss = running_loss / dset_sizes
                epoch_acc = running_corrects.double() / dset_sizes
                loss_all.append(int(epoch_loss * 100))
                acc_all.append(int(epoch_acc * 100))
                # print(epoch_loss)

                print('Loss: {:.4f} Acc: {:.4f}'.format(
                    epoch_loss, epoch_acc))

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model_ft.state_dict()
                if epoch_acc > 0.999:
                    break

    # save best model
    save_dir = data_dir + '/model'
    model_ft.load_state_dict(best_model_wts)
    model_out_path = save_dir + "/" + net_name + '_zyl.pth'
    if not os.path.exists(model_out_path):
        os.mkdir(model_out_path)
    else:
        torch.save(best_model_wts, model_out_path)

    # plot the figure of acc and loss
    x1 = list(range(len(acc_all)))
    x2 = list(range(len(loss_all)))
    y1 = acc_all
    y2 = loss_all
    plt.subplot(2, 1, 1)
    # plt.plot(x1, y1, 'o-',color='r')
    plt.plot(x1, y1, 'o-', label="Train_Accuracy")
    plt.title('train acc vs. iter')
    plt.ylabel('train accuracy')
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-', label="Train_Loss")
    plt.xlabel('train loss vs. iter')
    plt.ylabel('train loss')
    plt.legend(loc='best')
    plt.savefig(save_dir + "/" + "acc_loss.png")
    plt.show()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return train_loss, best_model_wts,model_ft



def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=10):
    """Decay learning rate by a f#            model_out_path ="./model/W_epoch_{}.pth".format(epoch)
#            torch.save(model_W, model_out_path) actor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.8**(epoch // lr_decay_epoch))
    print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
# train
pth_map = {
    'efficientnet-b0': 'efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'efficientnet-b7-dcc49843.pth',
    'efficientnet-b8': 'adv-efficientnet-b8-22a8fe65.pth',

}
# 离线加载预训练
model_ft = EfficientNet.from_name(net_name)
net_weight = './EfficientNet_model/' + pth_map[net_name]     #####################EfficientNet_model
state_dict = torch.load(net_weight)
model_ft.load_state_dict(state_dict)

# 修改全连接层
num_ftrs = model_ft._fc.in_features
model_ft._fc = nn.Linear(num_ftrs, class_num)

criterion = nn.CrossEntropyLoss()   #获得交叉熵损失
if use_gpu:
    model_ft = model_ft.cuda()
    criterion = criterion.cuda()
    
optimizer = optim.SGD((model_ft.parameters()), lr=lr,
                      momentum=momentum, weight_decay=0.0002)

train_loss, best_model_wts,model_ft= train_model(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)

# test
#model_ft.load_state_dict(best_model_wts)
#
#data_transforms = transforms.Compose([
#    transforms.Resize(350),
#    transforms.CenterCrop(350),
#    transforms.ToTensor(),
#    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#    ])
#
#
#def get_key(dct, value):
#    return [k for (k, v) in dct.items() if v == value]
#
#Species_id = []
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
## find the mapping of folder-to-label
#
#data = datasets.ImageFolder('/image/train')
#mapping = data.class_to_idx #别名与数字类别的映射关系字典（class_to_idx）
#print(mapping)
    
## start testing
#
#data_file = pd.read_csv('/af2020cv-2020-05-09-v5-dev/test.csv')
#File_id = data_file["FileID"].values.tolist()
#
#for i in range(len(File_id)):
#    test_dir = File_id[i] + '.jpg'
#    img_dir = '/image/test/'+test_dir  #读每一张测试照片
#
#    # load image
#    img = Image.open(img_dir)
#    inputs = data_transforms(img)
#    inputs.unsqueeze_(0)
#
#    if use_gpu:
#        model = model_ft.cuda() # use GPU
#    else:
#        model = model_ft
#    model.eval()
#    if use_gpu:
#        inputs = Variable(inputs.cuda()) # use GPU
#    else:
#        inputs = Variable(inputs)
#
#    # forward
#    outputs = model(inputs)
#    _, preds = torch.max(outputs.data, 1)
#    class_name = get_key(mapping, preds.item())
#    class_name = '%s' % (class_name)
#    class_name = class_name[2:-2]
#
#    print(img_dir)
#    print('prediction_label:', class_name)
#    print(30*'--')
#    Species_id.append(class_name)
#
#test = pd.DataFrame({'FileId':File_id,'SpeciesID':Species_id}) #将机器分类结果存储在.csv文件中
#test.to_csv('result.csv',index = None,encoding = 'utf8')




def test_model(model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=16, set_name='test', shuffle=False)
    for data in dset_loaders['test']:
        inputs, target = data
        target = torch.squeeze(target.type(torch.LongTensor))
        inputs, target = Variable(inputs.cuda()), Variable(target.cuda())

        with torch.no_grad():
            outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, target)

        if cont == 0:
            outPre = outputs.data.cpu()
            outLabel = target.data.cpu()
        else:
            outPre = torch.cat((outPre, outputs.data.cpu()), 0)
            outLabel = torch.cat((outLabel, target.data.cpu()), 0)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == target.data)
        cont += 1
    print('Loss: {:.4f} Acc: {:.4f}'.format(running_loss / dset_sizes,
                                            running_corrects.double() / dset_sizes))
# test
print('-' * 10)
print('Test Accuracy:')
model_ft.load_state_dict(best_model_wts)
criterion = nn.CrossEntropyLoss().cuda()
test_model(model_ft, criterion)
