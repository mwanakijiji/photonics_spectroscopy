import numpy as np

def toy_commands(num_cmds):

    # number of basis set commands
    # (equivalently, illuminations for every <lenslet,lambda> pair)
    n_cmds = num_cmds

    # make list of basis set commands
    # (note x and y don't nec. have to be the same)
    basis_set_cmds = np.zeros((n_cmds,10)) # initialize a blank command vector

    # convention here:
    # [ cmd_number , signal ]
    for i in range(0,n_cmds-1):

        basis_set_cmds[i,i] = 1 # only 1 'command' pixel is activated at any time

    # extra dimension, to allow NxM matrix which helps with troubleshooting
    # (just repeat the last command)
    basis_set_cmds[n_cmds-1,:] = basis_set_cmds[n_cmds-2,:]

    return basis_set_cmds