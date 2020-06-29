import torch
import os
import numpy as np

class BaseModel():
    def save_network(self, network, network_label, epoch_label, save_dir, on_gpu=True):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)

        if on_gpu:
            network.cuda()

    def load_network(self, network, network_label, epoch_label, save_dir):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))
        print('load network from ', save_path)

    def print_networks(self, net):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        #print(net)
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')