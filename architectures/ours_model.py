import timm
import torch.nn as nn
import pdb
import architectures.ours as ours



class Network(nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()
        self.model = ours.vit_small_patch16_224(pretrained = True)
        n_features = self.model.head.in_features
        # pdb.set_trace()
        self.model.head = nn.Linear(n_features, opt.embed_dim)
        self.name = opt.arch

    def forward(self, x):
        x = self.model(x)

        return nn.functional.normalize(x, dim=-1)
