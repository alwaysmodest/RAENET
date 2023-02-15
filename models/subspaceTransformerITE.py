# -*- coding: utf-8 -*-
"""
@Time    : 2020/12/29 10:28
@Author  : Eric
@Email   : handlecoding@foxmail.com
@File    : Modules.py
"""

import torch.utils.data
import torch
from torch import nn
import torch.nn.functional as F
from layers.Transformer_layers import *

def compute_D(embeddings):
    t1 = embeddings.unsqueeze(1).expand(len(embeddings), len(embeddings), embeddings.shape[1])
    t2 = embeddings.unsqueeze(0).expand(len(embeddings), len(embeddings), embeddings.shape[1])
    d = (t1 - t2).pow(2).sum(2)
    return d

class Encoder(nn.Module):
    def __init__(self, input_dims, hidden_dims):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)

    def forward(self, input):
        output = F.relu(self.fc1(input))
        embeddings = self.fc2(output)

        return embeddings


class TransformerEncoder(nn.Module):
    """Simplified implementation of the Vision transformer.

    Parameters
    ----------
    img_size : int
        Both height and the width of the image (it is a square).

    patch_size : int
        Both height and the width of the patch (it is a square).

    in_chans : int
        Number of input channels.

    n_classes : int
        Number of classes.

    embed_dim : int
        Dimensionality of the token/patch embeddings.

    depth : int
        Number of blocks.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer.

    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has `embed_dim` elements.

    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches.
        It has `(n_patches + 1) * embed_dim` elements.

    pos_drop : nn.Dropout
        Dropout layer.

    blocks : nn.ModuleList
        List of `Block` modules.

    norm : nn.LayerNorm
        Layer normalization.
    """

    def __init__(
            self,
            dim_of_data,
            down_out_dims,
            predict_dims,
            embed_dim=1,
            depth=3,
            n_heads=12,
            mlp_ratio=2.,
            qkv_bias=True,
            p=0.,
            attn_p=0.,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.down = torch.nn.Linear(dim_of_data, down_out_dims)
        self.predict = torch.nn.Linear(down_out_dims, predict_dims)
        self.ps=Propensity_net_NN(phase = "train")
        self.weight = torch.nn.Parameter(torch.rand(dim_of_data, down_out_dims), requires_grad=True)

    def forward(self, x: torch.Tensor):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, 1, dims)`.

        Returns
        -------
        logits : torch.Tensor
            Logits over all the classes - `(n_samples, n_classes)`.
        """
        x = torch.transpose(x, 1, 2)  # x:(n_samples, dims, 1)
        for block in self.blocks:
            x = block(x)
        x = torch.transpose(x, 1, 2)  # x:(n_samples, 1, dims)
        embedding = self.down(x)
        embedding = embedding.squeeze()
        embedding = torch.matmul(embedding, self.weight)
        embedding = embedding.unsqueeze(dim=1)
        prediction = self.ps(embedding)
        #print(prediction.shape)
        return embedding, prediction



class Decoder(nn.Module):
    def __init__(self, input_dims, out_dims):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(input_dims, out_dims)
        self.fc2 = nn.Linear(out_dims, out_dims)

    def forward(self, input):
        output = F.relu(self.fc1(input))
        output = self.fc2(output)

        return output


class AE(nn.Module):
    def __init__(self, input_dims, hid_dims, out_dims):
        super(AE, self).__init__()
        self.encoder = Encoder(input_dims, hid_dims)
        self.decoder = Decoder(hid_dims, out_dims)

    def forward(self, input):
        embeddings = self.encoder(input)
        x_reconstruct = self.decoder(embeddings)
        return x_reconstruct

class OutNet(nn.Module):
    def __init__(self, hid_dims, out_dims):
        super(OutNet, self).__init__()

        self.fc1 = nn.Linear(hid_dims, hid_dims)
        self.fc2 = nn.Linear(hid_dims, out_dims)

    def forward(self, input):
        return self.fc2(F.elu(self.fc1(input)))


class Discriminator(nn.Module):
    def __init__(self, input_dims, hid_dims, out_dims):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dims, hid_dims),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hid_dims, out_dims)
        )

    def forward(self, x):
        return self.model(x)


'''
    print(train_x.shape)  torch.Size([64, 30])
    print(train_y.shape)  torch.Size([64])
    print(train_t.shape)  torch.Size([64])
    print(input_x.shape)  torch.Size([64, 31])
'''

class SubTransCasual(nn.Module):
    def __init__(self,dim_of_data, down_out_dims, predict_dims, embed_dim, depth, n_heads):
        super(SubTransCasual, self).__init__()
        self.encoder = TransformerEncoder(dim_of_data=dim_of_data, down_out_dims=down_out_dims,
                                          predict_dims=predict_dims,
                                          embed_dim=embed_dim, depth=depth, n_heads=n_heads)
    def forward(self, x:torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        embedding,encoder_prediction = self.encoder(x)
        return embedding,encoder_prediction


# phase = ["train", "eval"]
class Propensity_net_NN(nn.Module):
    def __init__(self, phase):
        super(Propensity_net_NN, self).__init__()
        self.phase = phase
        self.fc1 = nn.Linear(in_features=30,out_features=30)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(in_features=30, out_features=30)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.ps_out = nn.Linear(in_features=30, out_features=2)

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.ps_out(x)
        if self.phase == "eval":
            return F.softmax(x, dim=1)
        else:
            return x

if __name__ == '__main__':
    '''
        dim_of_data: Any,
        down_out_dims: Any,
        predict_dims: Any,
        embed_dim: Any,
        depth: Any,
        n_heads: Any) -> None
    '''
    net = SubTransCasual(30, 15, 1, 1, 3, 3)
    x = torch.randn(64, 1, 30)
    embeddings, encoder_prediction, prediction = net(
        x)  # torch.Size([64, 1, 15]) torch.Size([64, 1, 1]) torch.Size([64, 1, 1])
    embeddings, encoder_prediction, prediction = embeddings.squeeze(), encoder_prediction.squeeze(), prediction.squeeze()
    print(embeddings.shape, encoder_prediction.shape, prediction.shape)

