import numpy as np
import matplotlib.pyplot as plt

results_loc = '/home/shic892/pyDNMFk/Results/nmf_test_over4_fall/5/'
data0 = np.load(results_loc+'W_reg_factors/W_0.npy')
#data0 = np.load(results_loc+'H_reg_factors/H_0.npy')
print(data0.shape)
data1 = np.load(results_loc+'W_reg_factors/W_1.npy')
#data1 = np.load(results_loc+'H_reg_factors/H_1.npy')
print(data1.shape)
data2 = np.load(results_loc+'W_reg_factors/W_2.npy')
#data2 = np.load(results_loc+'H_reg_factors/H_2.npy')
print(data2.shape)
data3 = np.load(results_loc+'W_reg_factors/W_3.npy')
#data3 = np.load(results_loc+'H_reg_factors/H_3.npy')
print(data3.shape)
data = np.concatenate((data0,data1,data2,data3), axis=0)
print(data.shape)
#print(data.sum(axis=0))
print(np.where(data[:,0]>0.005))