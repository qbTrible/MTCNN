import os

from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt

from simpling import FaceDataset


class Trainer:
    def __init__(self, net, save_path, dataset_path, isCuda=True):
        self.net = net
        self.save_path = save_path
        self.dataset_path = dataset_path
        self.isCuda = isCuda

        if self.isCuda:
            self.net.cuda()

        self.cls_loss_fn = nn.BCELoss()
        self.offset_loss_fn = nn.MSELoss()

        self.optimizer = optim.Adam(self.net.parameters(), weight_decay=0.00001)

        # if os.path.exists(self.save_path):
        #     net.load_state_dict(torch.load(self.save_path))

    def train(self):
        faceDataset = FaceDataset(self.dataset_path)
        dataloader = DataLoader(faceDataset, batch_size=1000, shuffle=True, num_workers=4)
        epochs = 0
        losses = []
        while True:
            for i, (img_data_, category_, offset_) in enumerate(dataloader):
                if self.isCuda:
                    img_data_ = img_data_.cuda()
                    category_ = category_.cuda()
                    offset_ = offset_.cuda()

                _output_category, _output_offset = self.net(img_data_)
                output_category = _output_category.view(-1, 1)
                output_offset = _output_offset.view(-1, 4)
                # output_landmark = _output_landmark.view(-1, 10)
                # 计算分类的损失
                category_mask = torch.lt(category_, 2)  # part样本不参与分类损失计算
                category = torch.masked_select(category_, category_mask)
                output_category = torch.masked_select(output_category, category_mask)
                cls_loss = self.cls_loss_fn(output_category, category)

                # 计算bound的损失
                offset_mask = torch.gt(category_, 0)  # 负样本不参与计算
                offset_index = torch.nonzero(offset_mask)[:, 0]  # 选出非负样本的索引
                offset = offset_[offset_index]
                output_offset = output_offset[offset_index]
                offset_loss = self.offset_loss_fn(output_offset, offset)  # 损失

                loss = cls_loss + offset_loss
                losses.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if i % 10 == 0:
                    print("epochs:", epochs, " batches:", i, " loss:", loss.cpu().data.numpy(), " cls_loss:", cls_loss.cpu().data.numpy(), " offset_loss",
                      offset_loss.cpu().data.numpy())
                    torch.save(self.net.state_dict(), self.save_path)
                    # plt.clf()
                    # plt.plot(losses)
                    # plt.xlabel("batches")
                    # plt.ylabel("losses")
                    # plt.pause(0.01)

                del img_data_, category_, offset_, _output_category, _output_offset, output_category, output_offset, category_mask, offset_index, offset, offset_loss, cls_loss, loss
            torch.save(self.net.state_dict(), self.save_path)
            # print("epochs {} save success".format(epochs))
            epochs += 1
