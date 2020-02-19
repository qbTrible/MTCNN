from nets01 import PNet
import train
import torch
import os
import time

if __name__ == '__main__':
    while True:
        try:
            net = PNet()
            if os.path.exists('./param01/pnet.pt'):
                net.load_state_dict(torch.load("./param01/pnet.pt"))

            trainer = train.Trainer(net, './param01/pnet.pt', r"F:\celeba1\12")
            trainer.train()
        except Exception as e:
            print(e)
            time.sleep(600)
    # net = PNet()
    # if os.path.exists('./param/pnet.pt'):
    #     net.load_state_dict(torch.load("./param/pnet.pt"))
    # trainer = train.Trainer(net, './param/pnet.pt', r"F:\celeba1\12")
    # trainer.train()