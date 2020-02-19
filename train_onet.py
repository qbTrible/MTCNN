from nets01 import ONet
import train
import os
import torch

if __name__ == '__main__':
    while True:
        try:
            net = ONet()
            if os.path.exists('./param01/onet.pt'):
                net.load_state_dict(torch.load("./param01/onet.pt"))

            trainer = train.Trainer(net, './param01/onet.pt', r"F:\celeba1\48")
            trainer.train()
        except Exception as e:
            print(e)
