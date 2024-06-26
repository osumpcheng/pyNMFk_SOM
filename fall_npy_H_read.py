import numpy as np
import matplotlib.pyplot as plt

results_loc = '/home/shic892/pyDNMFk/Results/fall_nmf_peaktable_over4_031923/7/'
data0 = np.load(results_loc+'H_reg_factors/H_0.npy')
print(data0.shape)
data1 = np.load(results_loc+'H_reg_factors/H_1.npy')
print(data1.shape)
data2 = np.load(results_loc+'H_reg_factors/H_2.npy')
print(data2.shape)
data3 = np.load(results_loc+'H_reg_factors/H_3.npy')
print(data3.shape)
data = np.concatenate((data0,data1,data2,data3), axis=1)
np.savetxt('fall_nmf_test_over4_trends_031923.csv', data, delimiter=',')
print(data.shape)

plt.figure(0)
plt.rc('font', family='serif')
for i in range(7):
    data_i = data[i,:]
    #data_i = data_i[np.argsort(data_i)]
    print(data_i.shape)    
    features = plt.plot(range(len(data_i)), data_i, 'o-')

plt.legend(["NMF1", "NMF2", "NMF3", "NMF4", "NMF5", "NMF6", "NMF7"])    
plt.savefig(results_loc+'fall_trends_all_over4_031923.png', dpi=300)
#print(data[1,0:10])