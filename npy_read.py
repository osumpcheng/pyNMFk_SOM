import numpy as np
import matplotlib.pyplot as plt

data0 = np.load('/tahoma/emsls60141/FT-ICR-MS/results/pynmfk/1000s_pynmfk_test/9/H_reg_factors/H_0.npy')
data1 = np.load('/tahoma/emsls60141/FT-ICR-MS/results/pynmfk/1000s_pynmfk_test/9/H_reg_factors/H_1.npy')
data2 = np.load('/tahoma/emsls60141/FT-ICR-MS/results/pynmfk/1000s_pynmfk_test/9/H_reg_factors/H_2.npy')
data3 = np.load('/tahoma/emsls60141/FT-ICR-MS/results/pynmfk/1000s_pynmfk_test/9/H_reg_factors/H_3.npy')
data = np.concatenate((data0,data1,data2,data3), axis=1)

for i in range(9):
    data_i = data[i,:]
    data_i = data_i[np.argsort(data_i)]
    print(data_i.shape)
    plt.figure(0)
    features = plt.scatter(range(len(data_i)), data_i, c ="blue")
    features.figure.savefig('feature_import_' + str(i) +'.png', dpi=300)
    plt.clf()

#print(data[1,0:10])