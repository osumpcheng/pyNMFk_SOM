import h5py

f = h5py.File('/home/shic892/pyDNMFk/Results/nmf_test_over1_fall_t/6/results.h5','r')
print(list(f.keys()))
data = f['ErrTol'] 
print(data.shape)
print(data[0:20])