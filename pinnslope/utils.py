import numpy as np
import torch
import random



def grid(ox, dx, nx, ot, dt, nt, normalization):
    """
    Creates grid points for PINN input.

    Parameters
    ----------
    ox : `int`
        origin of grid coordinates, x dimension
    dx : `int`
        trace spacing 
    nx : `int`
        total number of traces
    ot : `int`
        origin of grid coordinates, t dimension
    dt : `int`
        time samples interval
    nt : `int`
        total number of time samples
    normalization : `boolean`
        if 'True' applied normalization to grid points

    Returns
    -------
    x : `numpy.ndarray`
        1-D array with x coordinates
    t : `numpy.ndarray`
        1-D array with t coordinates
    X : `numpy.ndarray`
        coordinate matrix for x dimension (from 1D coordinate vector)
    T : `numpy.ndarray`
        coordinate matrix for t dimension (from 1D coordinate vector)
    grid : `numpy.ndarray`
        coordinate array
    """

    dim_x = int(nx-ox); dim_t = int(nt-ot) 
    
    x = ((np.arange(dim_x) + ox) * dx)
    t = ((np.arange(dim_t) + ot) * dt)
    
    if normalization==True:
        x = x/np.max(x)
        t = t/np.max(t)

    X, Z = np.meshgrid(x, t, indexing='ij')
    Xinp, Zinp = X.ravel(), Z.ravel()
    Xinp, Zinp = np.expand_dims(Xinp, axis=1), np.expand_dims(Zinp, axis=1)
    input = np.concatenate((Xinp,Zinp), axis=1)
    
    x, t = np.expand_dims(x, axis=1), np.expand_dims(t, axis=1)
    
    return x, t, X, Z, input




def grid_subsampling(Xmesh, Tmesh, Rop, n_of_training_traces):
    """
    It applies the restriction operator to the full grid. The resulting grid will contain only the grid-points
    corresponding to the position of the available traces needed to train the network.
    
    Parameters
    ----------
    Xmesh : `numpy.ndarray`
        coordinate matrix (from 1D coordinate vector)
    Tmesh : `numpy.ndarray`
        coordinate matrix (from 1D coordinate vector)
    Rop : `pylops object`
        Restriction operator
    n_of_training_traces : `int`
        total number of training traces

    Returns
    -------
    traces_training_grid : `numpy.ndarray`
        1D array containing only the grid-points corresponding to the position of the available traces 
        needed to train the network
    """
    
    Xmsh, Tmsh = Xmesh.copy(), Tmesh.copy()
    Xmsh, Tmsh = Rop*Xmsh.ravel(), Rop*Tmsh.ravel()
    Xmsh, Tmsh = Xmsh.reshape(n_of_training_traces, Xmesh.shape[1]), Tmsh.reshape(n_of_training_traces, Tmesh.shape[1])
    #Xmsh[~mask_obs.T]=np.NaN;    Tmsh[~mask_obs.T]=np.NaN
    Xmsh_resh, Tmsh_resh = Xmsh.ravel(), Tmsh.ravel()
    Xmsh_resh_filt, Tmsh_resh_filt = Xmsh_resh[~ np.isnan(Xmsh_resh)], Tmsh_resh[~ np.isnan(Tmsh_resh)]; 
    traces_training_grid = np.concatenate( ( np.expand_dims(Xmsh_resh_filt, axis=1), np.expand_dims(Tmsh_resh_filt, axis=1) ), axis=1 )
    
    return traces_training_grid




def training_traces(data_obs):
    """
    Training traces creation.

    Takes in input the data after the Restriction Operator (Rop) has been applied, 
    and reshapes the array in a 1D array compose of all the available traces one below the other.

    Parameters
    ----------
    data_obs : 'numpy.ndarray'
        seismic gather at which the pylops restriction operator has been applied

    Return
    ------
    ground_truth_traces : 'numpy.ndarray'
        1D array composed of all the subsampled traces placed them one below the other
         from the first one to the last one
    """
    d = data_obs.copy()
    ground_truth_traces = np.expand_dims( (d).T.ravel(), axis=1 )
    return ground_truth_traces






def mse(xref, xcmp):
    """Mean Square Error (MSE)

    Compute Mean Square Error between two vectors

    Parameters
    ----------
    xref : :obj:`numpy.ndarray`
        Reference vector
    xcmp : :obj:`numpy.ndarray`
        Comparison vector

    Returns
    -------
    mse : :obj:`float`
        Mean Square Error

    """
    mse = np.mean(np.abs(xref - xcmp) ** 2)
    return mse




def snr(xref, xcmp):
    """Signal to Noise Ratio (SNR)

    Compute Signal to Noise Ratio between two vectors

    Parameters
    ----------
    xref : :obj:`numpy.ndarray`
        Reference vector
    xcmp : :obj:`numpy.ndarray`
        Comparison vector

    Returns
    -------
    snr : :obj:`float`
        Signal to Noise Ratio of ``xcmp`` with respect to ``xref``

    """
    xrefv = np.mean(np.abs(xref) ** 2)
    snr = 10.0 * np.log10(xrefv / mse(xref, xcmp))
    return snr



def set_seed(seed):
    """Set all random seeds to a fixed value and take out any
    randomness from cuda kernels
    Parameters
    ----------
    seed : :obj:`int`
        Seed number
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True
