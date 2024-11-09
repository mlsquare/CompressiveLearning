import torch
import torch.nn as nn
from sklearn.random_projection import SparseRandomProjection
from scipy.stats import laplace  # Import Laplace distribution

class A2DLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        pdf='normal',
        sign_fn='tanh',
        cdf_fn='sigmoid',
        quantile_tx=False,
        trainable=True
    ):
        super(A2DLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.args = {
            'pdf': pdf,
            'sign_fn': sign_fn,
            'cdf_fn': cdf_fn,
            'quantile_tx': quantile_tx,
            'trainable': trainable
        }

        if sign_fn == 'tanh':
            self.sign = nn.Tanh()
        elif sign_fn == 'sign':
            self.sign = torch.sign
        else:
            self.sign = nn.Identity()

        if cdf_fn == 'sigmoid':
            self.cdf = nn.Sigmoid()
        else:
            self.cdf = nn.Identity()

        # Assign weights based on the pdf
        if pdf == 'normal':
            _projection_matrix = torch.randn(out_features, in_features)
        elif pdf == 'uniform':
            _projection_matrix = torch.rand(out_features, in_features)
        elif pdf == 'sparse':
            # Generate a sparse random matrix using scikit-learn
            sparse_matrix = SparseRandomProjection(out_features, in_features, density=0.1)
            _projection_matrix = torch.tensor(sparse_matrix.toarray(), dtype=torch.float32)
        elif pdf == 'laplace':
            # Generate weights from Laplace distribution
            laplace_weights = laplace.rvs(size=(out_features, in_features))
            _projection_matrix = torch.tensor(laplace_weights, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported pdf: {pdf}")

        # Assign weights explicitly
        self.linear.weight = nn.Parameter(_projection_matrix)

        # Sample offset term (bias) from uniform distribution (0,1)
        if quantile_tx:
            self.quantile_offset = nn.Parameter(torch.rand(out_features))
        else:
            self.quantile_offset = 0

        # Set requires_grad based on trainable flag
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.linear(x)
        x = self.cdf(x) - self.quantile_offset
        x = self.sign(x)
        return x