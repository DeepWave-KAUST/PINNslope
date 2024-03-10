import numpy as np
import torch
import torch.nn as nn


class PositionalEncod(nn.Module):
    """
    Anisotropic Positional Encoding (also, Fourier Feature Mapping) similar to the one in (cf. [1]).

    Linear sampling: k_i = pi*i

    gamma(input) = [ [ sin((k_i*input)/scale ), cos((k_i*input))/scale], ... ,
                   [ sin((k_L*input)/scale ), cos((k_L*input))/scale ] ]
    
    Parameters
    ----------
    PosEnc : `list` 
        list of integers values corresponding to: 
        - PosEnc[0] : `int` 
                number of frequency components along x-coordinate
        - PosEnc[1] : `int`
                number of frequency components along t-coordinate
        - PosEnc[2] : `int`
                scaling factor to lower the encoding frequencies (act on both x and t coord.)

    device : `string`
        name of the device to allocate the arrays of the Positional encoding operations, i.e ``cpu`` or if GPU is available ``cuda``.

    References
    ----------
    [1] Sun, Yu, et al. "Coil: Coordinate-based internal learning for imaging inverse problems." 
        arXiv preprint arXiv:2102.05181 (2021). 
    """

    def __init__(self, PosEnc=[1, 1, 2], device='cpu'):
        super().__init__()
        self.PEnc= PosEnc
        self.k_pi_x = (torch.tensor(np.pi)*torch.arange(self.PEnc[0])).reshape(-1, self.PEnc[0]).to(device); self.k_pi_x = self.k_pi_x.T
        self.k_pi_t = (torch.tensor(np.pi)*torch.arange(self.PEnc[1])).reshape(-1, self.PEnc[1]).to(device); self.k_pi_t = self.k_pi_t.T
    
    def forward(self, input):
        scale = self.PEnc[2]
        tmpx = torch.cat([torch.cos( self.k_pi_x*input[:,0]/scale ), torch.sin( self.k_pi_x*input[:,0]/scale )], axis=0)
        tmpt = torch.cat([torch.cos( self.k_pi_t*input[:,1]/scale ), torch.sin( self.k_pi_t*input[:,1]/scale )], axis=0)
        cat = torch.cat((tmpx, tmpt), axis=0)

        return cat.T



class Sin(nn.Module):
    """Define sin activation function"""
    
    def forward(self,input):
        return torch.sin(input)


def activation(act_fun='LeakyReLU'):
    """Easy selection of activation function by passing string or
    module (e.g. nn.ReLU)
    """
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'ReLU':
            return nn.ReLU()
        elif act_fun == 'Tanh':
            return nn.Tanh()
        elif act_fun == 'Sin':
            return Sin()
        else:
            raise ValueError(f'{act_fun} is not an activation function...')
    else:
        return act_fun()


class adaptivelinear(nn.Linear):
    
    def __init__(self, in_features, out_features, bias=True, adaptive_type='layer', adaptive_rate=None, adaptive_rate_scaler=None):
        super(adaptivelinear, self).__init__(in_features, out_features, bias)
        self.adaptive_type=adaptive_type
        self.adaptive_rate=adaptive_rate
        self.adaptive_rate_scaler=adaptive_rate_scaler
        
        if self.adaptive_rate:
            if self.adaptive_type == 'layer':
                self.adaptive = nn.Parameter(self.adaptive_rate * torch.ones(1))
            else:
                self.adaptive = nn.Parameter(self.adaptive_rate * torch.ones(self.out_features))
            if not self.adaptive_rate_scaler:
                self.adaptive_rate_scaler = 10.0
    
    
    def forward(self, input):
        if self.adaptive_rate:
            
            return self.adaptive_rate_scaler * self.adaptive * nn.functional.linear(input, self.weight, self.bias)
        return nn.functional.linear(input, self.weight, self.bias)


def layer(lay='linear'):
    """Easy selection of layer
    """
    if isinstance(lay, str):
        if lay == 'linear':
            return lambda x,y: nn.Linear(x, y)
        elif lay in ('adaptive', 'adaptive_neu'):
            return lambda x, y: adaptivelinear(x, y, adaptive_type='neuron', adaptive_rate=0.1, adaptive_rate_scaler=10.)
        elif lay == 'adaptive_lay':
            return lambda x,y: adaptivelinear(x,y, adaptive_type='layer', adaptive_rate=0.1, adaptive_rate_scaler=10.)
        else:
            raise ValueError(f'{lay} is not a layer type...')
    else:
        return lay


class NetworkPE(nn.Module):
    def __init__(self, n_input, n_output, n_hidden,
                 lay='linear', act='Tanh', PosEnc=[1,1,2], device='cpu'):
        super(NetworkPE, self).__init__()
        self.lay = lay
        self.act = act
        self.n_output = n_output
        act = activation(act)
        lay = layer(lay)
        self.PE = PositionalEncod(PosEnc, device=device)
        self.model = nn.Sequential(nn.Sequential(lay(n_input, n_hidden[0]), act),
                                   *[nn.Sequential(lay(n_hidden[i], n_hidden[i + 1]),
                                     act) for i in range(len(n_hidden) - 1)],
                                   lay(n_hidden[-1], n_output))

    def forward(self, x, PE=False):

        x = self.PE(x)
        x = self.model(x)
        return x
        