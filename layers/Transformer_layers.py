import torch
import torch.nn as nn


class Attention(nn.Module):
    """Attention mechanism.

    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.

    n_heads : int
        Number of attention heads.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    attn_p : float
        Dropout probability applied to the query, key and value tensors.

    proj_p : float
        Dropout probability applied to the output tensor.


    Attributes
    ----------
    scale : float
        Normalizing consant for the dot product.

    qkv : nn.Linear
        Linear projection for the query, key and value.

    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.

    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super(Attention, self).__init__()
        self.n_heads = n_heads

        # ************************************************
        self.origin_dim = dim  # 第三个维度, 结构性数据为1
        self.dim = dim * n_heads
        # ************************************************

        self.head_dim = self.dim // n_heads
        self.scale = self.head_dim ** -0.5

        # ************************************************
        self.updim = nn.Linear(self.origin_dim, self.dim, bias=True)
        # ************************************************

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(self.dim, self.origin_dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, dims, 1)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, dims, 1)`.
        """
        n_samples, n_tokens, _ = x.shape

        x = self.updim(x)  # (n_samples, dims, 1 * n_heads)
        qkv = self.qkv(x)  # (n_samples, dims, 3 * n_heads)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, dims, 3, n_heads, 1)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, dims, 1)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, 1, dims)
        dp = (
                     q @ k_t
             ) * self.scale  # (n_samples, n_heads, dims, dims)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, dims, dims)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, dims, 1)
        weighted_avg = weighted_avg.transpose(
            1, 2
        )  # (n_samples, dims, n_heads, 1)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, dims, n_heads)

        x = self.proj(weighted_avg)  # (n_samples, dims, 1)
        x = self.proj_drop(x)  # (n_samples, dims, 1)

        return x


class MLP(nn.Module):
    """Multilayer perceptron.

    Parameters
    ----------
    in_features : int
        Number of input features.

    hidden_features : int
        Number of nodes in the hidden layer.

    out_features : int
        Number of output features.

    p : float
        Dropout probability.

    Attributes
    ----------
    fc : nn.Linear
        The First linear layer.

    act : nn.GELU
        GELU activation function.

    fc2 : nn.Linear
        The second linear layer.

    drop : nn.Dropout
        Dropout layer.
    """

    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, dims, 1)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, dims, 1)`
        """
        x = self.fc1(
            x
        )  # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)

        return x


class Block(nn.Module):
    """Transformer block.

    Parameters
    ----------
    dim : int
        Embeddinig dimension.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect
        to `dim`.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization.

    attn : Attention
        Attention module.

    mlp : MLP
        MLP module.
    """

    def __init__(self, dim, n_heads, mlp_ratio=2.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
        )

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


if __name__ == "__main__":
    x = torch.randn(64, 1, 18)
    net = Block(18,3)
    # net = Block(dim_of_data=18, embed_dim=1, depth=3, n_heads=4, mlp_ratio=2, embedding_down_rate=2)
    out = net(x)
    print(x[0])
    print(out[0])
    print(out.shape)

