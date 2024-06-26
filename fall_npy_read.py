import numpy as np
import matplotlib.pyplot as plt

results_loc = '/home/shic892/pyDNMFk/Resultsnmf_test_over1_fall/4/'
data0 = np.load(results_loc+'H_reg_factors/H_0.npy')
data1 = np.load(results_loc+'H_reg_factors/H_1.npy')
data2 = np.load(results_loc+'H_reg_factors/H_2.npy')
data3 = np.load(results_loc+'H_reg_factors/H_3.npy')
data = np.concatenate((data0,data1,data2,data3), axis=1)

for i in range(4):
    data_i = data[i,:]
    data_i = data_i[np.argsort(data_i)]
    print(data_i.shape)
    plt.figure(0)
    features = plt.scatter(range(len(data_i)), data_i, c ="blue")
    features.figure.savefig(results_loc+'fall_feature_import_' + str(i) +'.png', dpi=300)
    plt.clf()

#print(data[1,0:10])