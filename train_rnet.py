from nets01 import RNet
import train
import os
import torch

if __name__ == '__main__':
    while True:
        try:
            net = RNet()
            if os.path.exists('./param01/rnet.pt'):
                net.load_state_dict(torch.load("./param01/rnet.pt"))

            trainer = train.Trainer(net, './param01/rnet.pt', r"F:\celeba1\24")
            trainer.train()
        except Exception as e:
            print(e)
