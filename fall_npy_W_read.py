import numpy as np
import matplotlib.pyplot as plt

results_loc = '/home/shic892/pyDNMFk/Results/fall_nmf_peaktable_over4_031923/7/'
data0 = np.load(results_loc+'W_reg_factors/W_0.npy')
print(data0.shape)
data1 = np.load(results_loc+'W_reg_factors/W_1.npy')
print(data1.shape)
data2 = np.load(results_loc+'W_reg_factors/W_2.npy')
print(data2.shape)
data3 = np.load(results_loc+'W_reg_factors/W_3.npy')
print(data3.shape)
data = np.concatenate((data0,data1,data2,data3), axis=0)
print(data.shape)

np.savetxt('nmf_fall_W_features_031923.csv', data, delimiter=',')
#for i in range(5):
#    data_i = data[:,i]
#    data_i = data_i[np.argsort(data_i)]
#    print(data_i.shape)
#    plt.figure(0)
#    features = plt.plot(range(len(data_i)), data_i, 'ro-')
#    plt.savefig(results_loc+'fall_feature_weights_' + str(i) +'.png', dpi=300)
#    plt.clf()

#print(data[1,0:10])
