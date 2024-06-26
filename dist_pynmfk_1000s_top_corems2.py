#To run this code please run `mpirun -n 4 python dist_pynmfk_fall_test.py` in command line.


import sys


import pyDNMFk.config as config

config.init(0)
from pyDNMFk.pyDNMFk import *
from pyDNMFk.utils import *
from pyDNMFk.dist_comm import *
from scipy.io import loadmat



def dist_nmfk_2d_rand_init_swim():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    p_r, p_c = 2, 2

    comms = MPI_comm(comm, p_r, p_c)
    comm1 = comms.comm
    rank = comm.rank
    size = comm.size
    args = parse()
    args.size, args.rank, args.comm, args.p_r, args.p_c = size, rank, comms, p_r, p_c
    args.row_comm, args.col_comm, args.comm1 = comms.cart_1d_row(), comms.cart_1d_column(), comm1
    rank = comms.rank
    args.fpath = '/home/shic892/corems/'
    args.fname = '1000_soils_btm_37242x61_022124_nt'
    args.ftype = 'csv'
    args.start_k = 2
    args.end_k = 40
    args.sill_thr = 0.6
    args.itr = 1000
    args.init = 'rand'
    args.noise_var = 0.016
    args.verbose = True
    args.norm = 'kl'
    args.method = 'mu'
    args.checkpoint = True
    args.precision = np.float32
    A_ij = data_read(args).read().astype(args.precision)
    args.results_path = '/home/shic892/pyDNMFk/Results/'
    nopt = PyNMFk(A_ij, factors=None, params=args).fit()
    assert nopt == 16


dist_nmfk_2d_rand_init_swim()

