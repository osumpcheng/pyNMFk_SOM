import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform, expon

df_sites_qc = pd.read_csv('/home/shic892/corems/1000_soils_124x7880_122122.csv', index_col=0)
site_depth = [x[5:8] for x in df_sites_qc.index.values]
df_sites_qc_top = df_sites_qc.iloc[[x=='TOP' for x in site_depth],:]
df_sites_qc_top = df_sites_qc_top.loc[:,df_sites_qc_top.sum()>0]
df_sites_qc_btm = df_sites_qc.iloc[[x=='BTM' for x in site_depth],:]
df_sites_qc_btm = df_sites_qc_btm.loc[:,df_sites_qc_btm.sum()>0]
df_env = pd.read_csv('/home/shic892/corems/1000s_env_w_biome_meta_v3.csv', index_col = 0,encoding='latin-1')
df_env.columns = [x[0:10] for x in df_env.columns]

df_env = df_env.iloc[:,np.append(range(0,28), range(56,61))]
df_env = df_env.drop(['TKN_pct'], axis=1)

df_env['Total_Carb'] = np.log10(df_env['Total_Carb']+1)
df_env['Total_Nitr'] = np.log10(df_env['Total_Nitr']+1)
df_env['C_to_N_rat'] = np.log10(df_env['C_to_N_rat']+1)

df_env_top = df_env.loc[df_sites_qc_top.index.values,:]
df_env_btm = df_env.loc[df_sites_qc_btm.index.values,:]

# df_env_top_pred = df_env_top.iloc[:,np.append(range(0,21), range(24,32))]
# df_env_top_pred = df_env_top_pred.fillna(0.)
# df_env_btm_pred = df_env_btm.iloc[:,np.append(range(0,21), range(24,32))]
# df_env_btm_pred = df_env_btm.fillna(0.)

# df_env_top_pred = df_env_top_pred.loc[:,['mean_GWC', 'Mn_mg_per_','CEC_Meq_pe',  'Total_Nitr','Total_Carb']]
# df_env_btm_pred = df_env_btm_pred.loc[:,['mean_GWC', 'Mg_Meq_per','Total_Base', 'CEC_Meq_pe',  'Total_Nitr','Total_Carb']]

results_loc = '/home/shic892/pyDNMFk/Results/1000s_btm_5515x61_nt/5/'
# '/home/shic892/pyDNMFk/Results/1000s_btm_5515x61_nt'
# '/home/shic892/pyDNMFk/Results/1000s_top_7312x63_nt
datah0 = np.load(results_loc+'H_reg_factors/H_0.npy')
print(datah0.shape)
datah1 = np.load(results_loc+'H_reg_factors/H_1.npy')
print(datah1.shape)
datah2 = np.load(results_loc+'H_reg_factors/H_2.npy')
print(datah2.shape)
datah3 = np.load(results_loc+'H_reg_factors/H_3.npy')
print(datah3.shape)
datah5 = np.concatenate((datah0,datah1,datah2,datah3), axis=1)
#np.savetxt('1000s_nmf_9_H_124.csv', data, delimiter=',')
print(datah5.shape)

X, y = np.log(datah5.T+1), np.log(df_env_btm['Respiratio'].iloc[:,1].values+1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


rng = np.random.RandomState(0)

clf = GradientBoostingRegressor(random_state=rng)

param_dist = {
    "loss": ['squared_error','absolute_error', 'huber'],
    "n_estimators": randint(50,3000),
    "max_depth": randint(3,61),
    "max_features": ['auto', 'sqrt', 'log2',2,3,4,5],
    "min_samples_split": randint(2, 10),
    "criterion": ['friedman_mse', 'squared_error'],
    "learning_rate": [0.0001, 0.001, 0.01, 0.1, 1.0],
    "ccp_alpha" : expon(scale=0.01)
}

rsh_btm = RandomizedSearchCV(
    estimator=clf, param_distributions=param_dist,random_state=rng, n_iter=1000,return_train_score=True, 
     cv=5, 
)
rsh_btm.fit(X_train, y_train)

print(rsh_btm.best_estimator_.score(X_test,y_test))

import joblib
joblib.dump(rsh_btm, '/home/shic892/corems/gradient_nmf_5_log_btm_respiration_log_v4.pkl')
