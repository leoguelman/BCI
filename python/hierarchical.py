# A Hierachical Model

X, z, y, a_effects, m_effects, i_effects = pre_process_data(df, standardize_x=False, interactions=True, 
                                                            p_score=pscore, drop_first=False)

s_effects = ['XC_0','XC_1','XC_2','XC_3','XC_4','X1', 
             'X2','X3','X4','X5']
ns_effects = ['S3','C2','C3','C1_1','C1_2','C1_3','C1_4',
              'C1_5','C1_6','C1_7','C1_8','C1_9','C1_10',
              'C1_11','C1_12','C1_13','C1_14','C1_15','p_score']
sz_effects = ['Z_XC_0','Z_XC_1','Z_XC_2','Z_XC_3','Z_XC_4','Z_X1',
              'Z_X2','Z_X3','Z_X4','Z_X5']
nsz_effects = ['Z_S3','Z_C2','Z_C3','Z_C1_1','Z_C1_2', 'Z_C1_3',
               'Z_C1_4','Z_C1_5','Z_C1_6','Z_C1_7','Z_C1_8',
               'Z_C1_9','Z_C1_10','Z_C1_11','Z_C1_12','Z_C1_13',
               'Z_C1_14','Z_C1_15','Z_p_score']

# Get indexes of each effect group
idx_s_effects = [a_effects.index(i) for i in s_effects]
idx_ns_effects = [a_effects.index(i) for i in ns_effects]
idx_sz_effects = [a_effects.index(i) for i in sz_effects]
idx_nsz_effects = [a_effects.index(i) for i in nsz_effects]

stan_data_mbi_h = {'N':      X.shape[0], 
                   'J':      len(df['schoolid'].unique()),
                   'school': df['schoolid'].values,
                   'N_s':    len(idx_s_effects),
                   'N_sz':   len(idx_sz_effects),
                   'N_ns':   len(idx_ns_effects),
                   'N_nsz':  len(idx_nsz_effects),
                   'x_s':    X[:,idx_s_effects],
                   'x_sz':   X[:,idx_sz_effects],
                   'x_ns':   X[:,idx_ns_effects],
                   'x_nsz':  X[:,idx_nsz_effects],
                   'y':      y,
                   'z':      z,
                   'rho':    0.0}
                   
sm = pystan.StanModel('../stan/stan_mbi_hierarchical.stan') 
multiprocessing.set_start_method("fork", force=True)
fit_mbi_h = sm.sampling(data=stan_data_mbi_h, iter=1000, chains=4, seed=194838)

