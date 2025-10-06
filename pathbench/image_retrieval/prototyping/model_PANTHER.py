# Model initiation for PANTHER

from torch import nn
import numpy as np
import os

from .PANTHER.layers import PANTHERBase
from ..utils import load_pkl

def check_prototypes(n_proto, embed_dim, load_proto, proto_path):
    """
    Check validity of the prototypes
    """
    if load_proto:
        assert os.path.exists(proto_path), "{} does not exist!".format(proto_path)
        if proto_path.endswith('pkl'):
            prototypes = load_pkl(proto_path)['prototypes'].squeeze()
        elif proto_path.endswith('npy'):
            prototypes = np.load(proto_path)


        assert (n_proto == prototypes.shape[0]) and (embed_dim == prototypes.shape[1]),\
            "Prototype dimensions do not match! Params: ({}, {}) Suplied: ({}, {})".format(n_proto,
                                                                                           embed_dim,
                                                                                           prototypes.shape[0],
                                                                                           prototypes.shape[1])
class PANTHER(nn.Module):
    """
    Wrapper for PANTHER model
    """
    def __init__(self, config, mode):
        super(PANTHER, self).__init__()

        self.config = config
        emb_dim = config.in_dim

        self.emb_dim = emb_dim
        self.heads = config.heads
        self.outsize = config.out_size
        self.load_proto = config.load_proto
        self.mode = mode

        check_prototypes(config.out_size, self.emb_dim, self.load_proto, config.proto_path)
        # This module contains the EM step
        self.panther = PANTHERBase(self.emb_dim, p=config.out_size, L=config.em_iter,
                         tau=config.tau, out=config.out_type, ot_eps=config.ot_eps,
                         load_proto=config.load_proto, proto_path=config.proto_path,
                         fix_proto=config.fix_proto)

    def representation(self, x):
        """
        Construct unsupervised slide representation
        """
        out, qqs = self.panther(x)
        return {'repr': out, 'qq': qqs}

    def forward(self, x):
        out = self.representation(x)
        return out['repr']
    
    """
    def predict(self, data_loader, use_cuda=True):
        if self.mode == 'classification':
            output, y = predict_clf(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'survival':
            output, y = predict_surv(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'emb':
            output = predict_emb(self, data_loader.dataset, use_cuda=use_cuda)
            y = None
        else:
            raise NotImplementedError(f"Not implemented for {self.mode}!")
        
        return output, y"""