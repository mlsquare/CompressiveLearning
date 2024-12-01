import torch
import torch.nn as nn

from omegaconf import OmegaConf
import itertools
import numpy as np

from ista import ista

amm_common_params = OmegaConf.load('commons.yaml')
amm_layer_params = OmegaConf.load('layer-01-test.yaml')

def get_set_sizes(N, p):
    p = np.array(p)
    p = p / p.sum()
    sizes = np.floor(N * p).astype(int).tolist()
    difference = N - sum(sizes)
    for i in range(difference):
        sizes[i % len(sizes)] += 1
    return sizes


def get_amm_parameters(dims,amm_layer_params = amm_layer_params.copy(), default_amm_params = amm_common_params.copy()):

    header = amm_layer_params[0]['header']
    num_tables = header['num_tables']
    table_params = amm_layer_params[1:]
    table_prop = [d['prop'] for d in table_params] 
    
    # convention y = xW (to be consistent with PyTorch)
    n = dims[0]
    m = dims[1]
    header['x_dim'] = n
    header['y_dim'] = m
    
    # contiguous blocks
    num_components =  min(m,n)
    sizes = get_set_sizes(num_components,table_prop)
    start_indexes = [0]
    start_indexes.extend(sizes[:-1])
    end_indexes = list(itertools.accumulate(sizes))
    
    header['start_indexes'] = start_indexes
    header['end_indexes'] = end_indexes

    amm_layer_params[0]['header'] = header
    
    for i in range(num_tables):
        table_param = table_params[i]

        # if this table has zero length, drop it
        if sizes[i] ==0:
            table_param['drop'] = True

        key_encoder = table_param['key_encoder']
        key_encoder = {**default_amm_params, **key_encoder}
        key_encoder['input_dim'] = n
        header['x_dim'] = n

        val_encoder = table_param['val_encoder']
        val_encoder = {**default_amm_params, **val_encoder}
        val_encoder['input_dim'] = m

        
        table_param['key_encoder'] = key_encoder
        table_param['val_encoder'] = val_encoder
        amm_layer_params[i+1] = table_param
        
    return amm_layer_params



def get_embedding_matrix(params):

    
    # sample from a family of distribution
    # to do; appropriately scale

    n = params['input_dim']
    d = params['embed_dim']
    dist = params['dist']
    ortho = params['ortho']
    bias = params['bias']
    normalize = params['normalize']
    scale = params['scale']
    gate = params['gate']

    if dist =='identity':
        A = torch.eye(n)
        return A
    
    if dist =='laplace':
        dist = torch.distributions.laplace.Laplace(0, 1)
    elif dist =='bernoulli':
        dist = torch.distributions.bernoulli.Bernoulli(0.5)
    elif dist =='uniform':
        dist = torch.distributions.uniform.Uniform(-1,1)
    else:
        dist = torch.distributions.normal.Normal(0,1)

    if ortho:
        s = max(d,n)
        tmp = dist.sample((s,s))
        Q = torch.linalg.qr(tmp)[0]
        A = Q[:n,:d]
    else:
        A = dist.sample((n,d))
    
    # normalize for unit-norm
    if bias or normalize:
        A = torch.nn.functional.normalize(A,dim=-1)

    # if bias (via quantile planes)    
    if bias:
        dist = torch.distributions.beta.Beta(1,1)
        b = dist.sample((size[1],))
        A += b

    if scale is not None:
        A = scale*A

    if gate == "tanh":
        A = torch.tanh(A)
    elif gate == "sigmoid":
        A = torch.sigmoid(A)
    else:
        pass

    return A


class AMM(nn.Module):
    def __init__(self, weight):
        super().__init__()
        
        dims = weight.shape
        params = get_amm_parameters(dims)

        header = params[0]['header']
        table_params = params[1:]
       
       
        num_tables = header['num_tables']
        start_indexes = header['start_indexes']
        end_indexes = header['end_indexes']

        # under y = xW
        V, S, U = torch.linalg.svd(weight, full_matrices=False)
        U = U.T
        
        # n: input_dim, m: output_dim, r: num_components = min(n,m)
        # V: n x r
        # U: m x r
        # S: r x 1
        # under y = Wx
        # U, S, V = torch.linalg.svd(weight, full_matrices=False)
        
        
        Key_Encoders = []
        Val_Encoders = []
        
        Keys = []
        Values = []
        Scales = []
        
        for ind in range(num_tables):
            table_param = table_params[ind]
            start_ind = start_indexes[ind]
            end_ind = end_indexes[ind]
            
            if table_param['drop']:
                # ignore the SVD factors
                Key_Encoder = None
                Val_Encoder = None

                local_keys = None
                local_vals = None
                local_scales = None

            if table_param['knn']:
            
                Key_Encoder = get_embedding_matrix(table_param['key_encoder']) 
                Val_Encoder = get_embedding_matrix(table_param['val_encoder'])

                local_keys = torch.matmul(V[:,start_ind:end_ind].T,Key_Encoder)
                local_vals = torch.matmul(U[:,start_ind:end_ind].T,Val_Encoder)
                local_scales = S[start_ind:end_ind].view(-1,1)
                
            else:
                # perform svm in full-precision

                Key_Encoder = None
                Val_Encoder = None

                local_keys = V[:,start_ind:end_ind]
                local_vals = U[:,start_ind:end_ind]
                local_scales = S[start_ind:end_ind].view(-1,1)

            Keys.append(local_keys)
            Values.append(local_vals)
            Scales.append(local_scales)
    
            Key_Encoders.append(Key_Encoder)
            Val_Encoders.append(Val_Encoder)
            
        self.params = params
        self.header = params[0]['header']
        self.table_params = params[1:]
       

        self.Keys = Keys
        self.Values = Values
        self.Scales = Scales
        self.Key_Encoders = Key_Encoders
        self.Val_Encoders = Val_Encoders
    
    
    def forward(self, x):
        
        header = self.header
        table_params = self.table_params
        num_tables = header['num_tables']   

        batch_dim = x.shape[0]
        y_dim = header['y_dim']
        y = torch.zeros((batch_dim,y_dim))
        
        for i in range(num_tables):
            # encode the query
            table_param = table_params[i]
            if table_param['drop']:
                continue
            if table_param['knn']:
                q = torch.matmul(x,self.Key_Encoders[i])
                alphas = torch.matmul(q,self.Keys[i].T)
                betas = alphas*self.Scales[i].T.repeat(batch_dim,1)
                print('betas',betas.shape)
                print('Values',self.Values[i].shape)
                yb = torch.matmul(betas,self.Values[i])
                # loop through each batch
                for b in range(batch_dim):
                    print('in fista for b',b)
                    yi = yb[b,:].reshape((1,yb.shape[1]))
                    z0 = torch.matmul(yi,self.Val_Encoders[i].T)
                    yh = ista(yi,z0, self.Val_Encoders[i].T, alpha=1.0, lr=False, maxiter=5)
                    y[b,:] = yh
            else:
                # perform low rank apprx
                print('in low-rank')
                print('K shape',self.Keys[i].shape)
                print('query shape',x.shape)
                alphas = torch.matmul(x,self.Keys[i])
                betas = alphas* self.Scales[i].T.repeat(batch_dim,1)
                y += torch.matmul(betas,self.Values[i].T)
        return y


W = torch.rand((30,20))
q = torch.rand((5,30))
amm = AMM(W)
yh = amm(q)
y = torch.matmul(q,W)