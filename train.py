import torch
import argparse
import tqdm
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import copy
import os
import pandas as pd

from dataloader import dataloder
from model import classifer
from utils import timeSince, create_dir

"""超参数部分"""
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--bs', default=32, type=int, help='在train集上的batchSize')
parser.add_argument('--lr', default=1e-4, type=float, help='')
parser.add_argument('--gpu', default=0, type=float, help='')
parser.add_argument('--model', default="resnet34", type=str, help='')
parser.add_argument('--scheduler', default="linear")
opt = parser.parse_args()


class Classifier_Trainer(object):
    def __init__(self):
        super(Classifier_Trainer, self).__init__()
        self.bs = opt.bs
        self.epoch = 30
        self.lr = opt.lr
        self.train_dataloader = dataloder(batch_size=self.bs, mode="train")
        self.test_dataloader = dataloder(batch_size=self.bs, mode="test")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 计算设备
        self.model = self.set_model(model_type=opt.model)
        self.optimer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)  # 优化器 负责更新参数
        self.set_scheduler()  # 负责调节学习率
        self.CE_weight = torch.FloatTensor([0.473, 0.527]).to(self.device)  # 0表示女性 1表示男性  女性占比0.527 男性占比0.473
        self.startTime = 0
        self.create_model_dir()

    def set_model(self, model_type="resnet34"):
        model = classifer(model_type)
        model = model.to(self.device)
        return model

    def set_scheduler(self):
        if opt.scheduler == 'None':
            self.scheduler = None
        elif opt.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimer, T_mult=2, T_0=5)
        elif opt.scheduler == 'linear':
            self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimer, step_size=1, gamma=0.9)

    def set_input(self, datas):
        images, labels = datas
        images = images.to(self.device)
        labels = labels.to(self.device)
        return images, labels

    def create_model_dir(self):
        name = os.path.join('checkpoints', opt.model)
        create_dir(name)
        self.checkpoints_dir = name

    def train(self, epoch_id):
        self.model.train()
        for batch_idx, datas in tqdm.tqdm(enumerate(self.train_dataloader)):
            images, labels = self.set_input(datas)
            pre = self.model(images)
            loss = F.cross_entropy(input=pre, target=labels, weight=self.CE_weight)
            # 梯度清零
            self.optimer.zero_grad()
            # 误差回传
            loss.backward()
            # 更新参数
            self.optimer.step()
            # 记录loss
            if (batch_idx + 1) % (len(self.train_dataloader) // 3) == 0:
                # 终端输出
                print("Epoch: %d/%d,batch_idx:%d,time:%s,Loss:%.5f" % (
                    epoch_id + 1, self.epoch, batch_idx + 1, timeSince(self.startTime), loss.data))

    def val(self):
        self.model.eval()
        accuracy = list()
        with torch.no_grad():
            for batch_idx, datas in tqdm.tqdm(enumerate(self.test_dataloader)):
                images, labels = self.set_input(datas)
                pred_list = self.model(images)
                for i in range(len(pred_list)):
                    pre = np.argmax(pred_list[i].data.cpu().numpy())
                    true = labels[i].item()
                    accuracy.append(pre == true)
        acc = sum(accuracy) / len(accuracy)
        print("在验证集上的正确率: %.4f" % acc)
        return acc

    def start(self, model_path=None):
        best_acc = 0
        # 如果有参数文件
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            print("加载参数文件: {}".format(model_path))
        self.startTime = time.time()
        for epoch in range(self.epoch):
            print("train............")
            torch.cuda.synchronize(self.device)
            self.train(epoch)
            print('val..............')
            acc = self.val()
            # 学习率策略
            old_lr = self.optimer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.optimer.param_groups[0]['lr']
            print("学习率: %s >>>> %s" % (old_lr, new_lr))
            if acc > best_acc:
                best_acc = acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                model_save_path = os.path.join(self.checkpoints_dir, '%.4f.pth' % (acc))
                torch.save(best_model_wts, model_save_path)

    def test(self, model_path=None):
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            print("加载参数文件: {}".format(model_path))
        self.true_test_dataloder = dataloder(batch_size=self.bs, mode="test", true_test=True)
        self.model.eval()
        image_id_list = []
        is_male_list = []
        with torch.no_grad():
            for batch_idx, datas in tqdm.tqdm(enumerate(self.true_test_dataloder)):
                image_name, images = datas
                images = images.to(self.device)

                pred_list = self.model(images)
                for i in range(len(pred_list)):
                    pre = np.argmax(pred_list[i].data.cpu().numpy())
                    if pre == 0:
                        pre = -1
                    image_id_list.append(image_name[i])
                    is_male_list.append(pre)
        outDict = {
            "image_id": image_id_list,
            "is_male": is_male_list
        }
        submit_data = pd.DataFrame(outDict)
        submit_data.to_csv("submission.csv", index=False)
        print("test完成!")


trainer = Classifier_Trainer()
trainer.start()

# 在测试集上测试
path = None
trainer.test(model_path=path)
