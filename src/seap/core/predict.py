import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# Handle both relative imports (when used as module) and absolute imports (when run directly)
#try:
#    from seap.prediction.sph_harm import quantum_number, orb_symbol
#except ImportError:
    # Add the src directory to the path for direct execution
if __package__ in (None, ""):
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from seap.prediction.sph_harm import quantum_number, orb_symbol

# This script is used to predict the coefficients of real spherical harmonics using different methods.
# The script supports three modes: neural network (nn), lasso regression (lr), and integration (itg).
# It also allows for optional arguments to specify the neural network ID, radius for lasso regression,
# and optimization of the radius. 
# The script also allows for outputting the results to a CSV file with a new method.
# The main function calls the appropriate function based on the specified mode.

def main(mode='nn', nnid=None, lrad=None, optr=False):
    """
    Main function to estimate coefficients of real spherical harmonics.

    Parameters
    ----------
    mode : str, optional
        Estimation method, by default 'nn'. Options are 'nn' (neural network), 
        'lr' (lasso regression), or 'itg' (integration).
    nnid : int, optional
        Identification number for a deep learning model, by default None.
    lrad : float, optional
        Radius for lasso regression or integration, by default None.
    optr : bool, optional
        Flag to optimize radius, by default False.

    Returns
    -------
    None
    """
    n = 4
    nx = 32
    ny = 32
    nz = 32
    psi_name = 'image32x32x32.npy'
    psi_info = 'image_info.npy'
    psi_dict = {'name': psi_name, 'info': psi_info, 'resolution': [nx, ny, nz]}
    
    if mode == 'nn':
        if nnid is None:
            nnid = 1
        deep_learning_model(n, nnid, **psi_dict)
    else:
        if not (mode == 'lr' or mode == 'itg'):
            sys.exit("mode must be either 'nn', 'lr', or 'itg'.")
        num_t = 64
        num_p = 64
        if mode == 'lr':
            sparse_modeling(n, num_t, num_p, lrad, optr, **psi_dict)
        if mode == 'itg':
            direct_calculation(n, num_t, num_p, lrad, optr, **psi_dict)

def deep_learning_model(n, nnid, **psi_dict): 
    """
    Estimating coefficients of real spherical harmonics using neural network.

    Parameters
    ----------
    n : int
        Number of coefficients.
    nnid : int
        Identification number for the neural network model.
    psi_dict : dict
        Dictionary containing information about the input data.

    Returns
    -------
    None
    """
    import importlib
    import torch
    from seap.prediction.datasets import BoxData
    
    # Load the encoder model
    path = os.path.join(os.path.dirname(__file__), f'../../../models/encoder_{nnid}')
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../models'))
    encoder = importlib.import_module(f'encoder_{nnid}.encoder')
    size = encoder.size
    n_div = encoder.n_div
    l_max = encoder.l_max
    rn_max = encoder.rn_max
    device = encoder.device

    if (n != l_max + 1):
        sys.exit("n and (l_max + 1) must be equal.")

    # Normalize the input data
    data = np.load(psi_dict.get('name'))
    xs_raw = [d.astype(np.float32) for d in data]
    factor = float(size / n_div)**3
    xsnorm = [np.square(x_raw).sum() * factor for x_raw in xs_raw]
    xs_mod = np.array([x_raw / np.sqrt(norm) for x_raw, norm in zip(xs_raw, xsnorm)])

    # Set the model in evaluation mode
    nc = (l_max + 1)**2
    model = encoder.NeuralNetwork(n_div**3, nc + rn_max + 1)
    model.load_state_dict(torch.load(f'{path}/encoder.pth', map_location='cpu'))
    model.eval()

    # Prediction
    sparams = []
    rparams = []
    mseloss = []
    data = BoxData(l_max, rn_max, n_div, size)
    with torch.no_grad():
        xs_torch = torch.from_numpy(xs_mod)
        for idx, x_torch in enumerate(xs_torch):
            x_torch.to(device)
            y = model(x_torch).to('cpu').detach().numpy()
            c, gamma, an = np.split(y, [nc, nc + 1])

            x_pred = np.ravel(data.params_to_boxdata(c.reshape(1, -1), gamma, an))
            x_orig = x_torch.to('cpu').detach().numpy()
            x_loss = np.average((x_pred - x_orig)**2)
            mseloss.append(x_loss)
            sparams.append(c)
            # sparams.append(c / np.linalg.norm(c))
            rparams.append([gamma, an])
    
    # Output results
    psi_info = np.load(psi_dict.get('info'))
    lmset = quantum_number(l_max + 1) 
    # output_csv(psi_info, lmset, sparams, orb_symbol)
    output_csv_new(psi_info, lmset, sparams, orb_symbol)

    oinfo = f'Target data : image32x32x32.npy\n\n'
    for info, sparam, rparam, loss in zip(psi_info, sparams, rparams, mseloss):
        oinfo += f'molecule {int(info[0]):>4d}, band {int(info[1]):>4d}, MSE {loss:>15.7E}\n\n'
        oinfo += 'predicted coefficients of real spherical harmonics:\n\n'
        for c, lm in zip(sparam, lmset):
            oinfo += f'l = {lm[0]:>2}, m = {lm[1]:>2} ({orb_symbol[(lm[0], lm[1])]:>12}): {c:>15.7E}\n'
        oinfo += '\n'
        oinfo += 'predicted parameters of radial function:\n\n'
        gamma, an = rparam
        oinfo += f'gamma [e^(-gamma*r)] = {gamma[0]:>15.7E}\n'
        for n, a in enumerate(an):
            oinfo += f'a{n}    [a{n}*r^({n})]     = {a:>15.7E}\n'
        oinfo += '\n'
    with open('nn.out', 'w') as fw:
        fw.write(oinfo) 
 
def sparse_modeling(n, num_t, num_p, lrad, optr, **psi_dict):
    """
    Estimating coefficients of real spherical harmonics using Lasso regression.

    Parameters
    ----------
    n : int
        Number of coefficients.
    num_t : int
        Number of time steps.
    num_p : int
        Number of parameters.
    lrad : float
        Radius for lasso regression.
    optr : bool
        Flag to optimize radius.
    psi_dict : dict
        Dictionary containing information about the input data.

    Returns
    -------
    None
    """
    from seap.prediction.lasso import LassoRegression
    lr = LassoRegression(num_t, num_p, n, **psi_dict)
    if (optr):
        opt_r_list, opt_alpha_list, clm_list = lr.run(rstep=0.05)
    else:
        clm_list, opt_alpha_list = lr.run_at_r(lrad)
    output_csv_new(lr.psi_info, lr.lmset, clm_list, orb_symbol)

def direct_calculation(n, num_t, num_p, lrad, optr, **psi_dict):
    """
    Estimating coefficients of real spherical harmonics using integration.

    Parameters
    ----------
    n : int
        Number of coefficients.
    num_t : int
        Number of time steps.
    num_p : int
        Number of parameters.
    lrad : float
        Radius for integration.
    optr : bool
        Flag to optimize radius.
    psi_dict : dict
        Dictionary containing information about the input data.

    Returns
    -------
    None
    """
    from seap.prediction.integration import Integration
    itg = Integration(num_t, num_p, n, **psi_dict)
    if (optr):
        opt_r_list, clm_list = itg.run()
    else:
        clm_list = itg.run_at_r(lrad)
    output_csv_new(itg.psi_info, itg.lmset, clm_list, orb_symbol) 

def output_csv(info_list, lm_list, sparams, orbsymbol):
    """
    Output the results to a CSV file.

    Parameters
    ----------
    info_list : list
        List of information about the molecules and bands.
    lm_list : list
        List of quantum numbers.
    sparams : list
        List of spherical harmonic coefficients.
    orbsymbol : dict
        Dictionary mapping quantum numbers to orbital symbols.

    Returns
    -------
    None
    """
    data = {'band_index':[], 'molecule_index':[], 'orbital':[], 'cval':[]}
    for info, sparam in zip(info_list, sparams):
        data['band_index'].append(int(info[1]))   
        data['molecule_index'].append(int(info[0]))
        index = np.argmax(np.abs(sparam))
        lm  = lm_list[index]
        data['orbital'].append(orbsymbol[(lm[0], lm[1])])
        data['cval'].append(sparam[index])
    df = pd.DataFrame(data)
    df.to_csv('orbital.csv', index=False)

def output_csv_new(info_list, lm_list, sparams, orbsymbol):
    """
    Output the results to a CSV file with a new method.

    Parameters
    ----------
    info_list : list
        List of information about the molecules and bands.
    lm_list : list
        List of quantum numbers.
    sparams : list
        List of spherical harmonic coefficients.
    orbsymbol : dict
        Dictionary mapping quantum numbers to orbital symbols.

    Returns
    -------
    None
    """
    data = {'band_index':[], 'molecule_index':[], 'orbital':[], 'cval':[]}

    # Grouping by band    
    tmp_band_indices = defaultdict(list)
    for i, bi in enumerate(info_list[:, 1]):
        tmp_band_indices[bi].append(i)
    band_indices = list(tmp_band_indices.values())

    # Determine the projection (orbital) sequentially, starting from
    # the band with the lowest energy 
    for indices in band_indices:
        clms = []
        molecules = []
        for i in indices:
            molecules.append(int(info_list[i, 0]))
            clms.append(sparams[i])

        # Under the condition that the projections do not overlap with
        # those of other bands, we adopt the orbital corresponding to 
        # the maximum clm as the projection.
        clms = np.array(clms)
        flat_clms = clms.flatten()
        for i in range(1, len(flat_clms)):
            ith_largest_clm = np.sort(np.abs(flat_clms))[-i]
            im, ilm = np.argwhere(np.abs(clms) == ith_largest_clm)[0]
            lm = lm_list[ilm]
            print(f'{np.sort(np.abs(flat_clms))}')
            print(f'{i}: clm = {ith_largest_clm}')
            print(f'mol:{molecules[im]}, orb:{orbsymbol[(lm[0], lm[1])]}')
            
            # Check if the same orbital of the same molecule already exists
            flag = 0
            for jm, jlm in zip(data['molecule_index'], data['orbital']):
                if jm == molecules[im] and jlm == orbsymbol[(lm[0], lm[1])]:
                    flag = 1
            if flag == 0:
                break

        data['band_index'].append(int(info_list[indices[0], 1]))   
        data['molecule_index'].append(molecules[im])
        data['orbital'].append(orbsymbol[(lm[0], lm[1])])
        data['cval'].append(clms[im, ilm])
        
    df = pd.DataFrame(data)
    df.to_csv('orbital.csv', index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='predict.py')
    parser.add_argument('--mode', dest='mode', default='nn', type=str, 
                        help='Estimation method, nn: neural_net, lr: lasso, itg:integration')
    parser.add_argument('--nnid', dest='nnid', default=1, type=int, 
                        help='Identification number for a deep learning model')
    parser.add_argument('--lrad', dest='lrad', default=0.2, type=float, 
                        help='radius (< 0.5)')
    parser.add_argument('--optr', action='store_true')
    args = parser.parse_args()
    main(mode=args.mode, nnid=args.nnid, lrad=args.lrad, optr=args.optr)
