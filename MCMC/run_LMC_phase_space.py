import numpy as np
import pickle
import gp_model as gpm

import os
print("OMP:", os.getenv("OMP_NUM_THREADS"))
print("MKL:", os.getenv("MKL_NUM_THREADS"))
print("OPENBLAS:", os.getenv("OPENBLAS_NUM_THREADS"))
print("numexpr:", os.getenv("NUMEXPR_NUM_THREADS"))

with open('./stripped_summary.pkl', 'rb') as f:
    mtots, mdms, mstars, mratios, peris, e_ratios, am_ratios, f_stripped_stars, f_stripped_DM, \
    f_stripped_stars_ICL, f_stripped_DM_ICL, e_ratios_ICL, am_ratios_ICL, \
    e_DM_mean, e_stars_mean, am_DM_mean, am_stars_mean, \
    e_DM_mean_ICL, e_stars_mean_ICL, am_DM_mean_ICL, am_stars_mean_ICL = pickle.load(f)

peri_orbits = np.asarray([4, 35, 90, 185, 445])
circ_orbits = np.asarray([0.11930828643367211, 0.3417820246165659, 0.5257561772547018, 0.6987225035818878, 0.9035057543970304])

circs = [circ_orbits[peri_orbits == peri_i][0] for peri_i in peris]
pick = np.asarray(mstars) > 1

def process_data(M_star, eta, e_dm, e_star, h_dm, h_star, f_strip_dm, f_strip_star):
    Y = np.column_stack([np.log10(e_dm), np.log10(h_dm), np.log10(f_strip_dm),
                         np.log10(e_star), np.log10(h_star), np.log10(f_strip_star)])
    X = np.column_stack([np.log10(M_star), eta])
    return X, Y

M_star_arr = np.array(mstars, dtype=np.float64)[pick]
eta_arr    = np.array(circs, dtype=np.float64)[pick]
e_star_arr = np.array(e_stars_mean, dtype=np.float64)[pick]
e_dm_arr   = np.array(e_DM_mean, dtype=np.float64)[pick]
h_star_arr = np.array(am_stars_mean, dtype=np.float64)[pick]
h_dm_arr   = np.array(am_DM_mean, dtype=np.float64)[pick]
f_strip_star = np.array(f_stripped_stars, dtype=np.float64)[pick]
f_strip_dm   = np.array(f_stripped_DM, dtype=np.float64)[pick]

X, Y = process_data(M_star_arr, eta_arr, e_dm_arr, e_star_arr, h_dm_arr, h_star_arr, f_strip_dm, f_strip_star)

e_star_all_icl = np.array(e_stars_mean_ICL, dtype=np.float64)[pick]
e_dmr_all_icl   = np.array(e_DM_mean_ICL, dtype=np.float64)[pick]
h_star_all_icl = np.array(am_stars_mean_ICL, dtype=np.float64)[pick]
h_dmr_all_icl   = np.array(am_DM_mean_ICL, dtype=np.float64)[pick]
f_strip_star_icl = np.array(f_stripped_stars_ICL, dtype=np.float64)[pick]
f_strip_dmr_icl   = np.array(f_stripped_DM_ICL, dtype=np.float64)[pick]

X_icl, Y_icl = process_data(M_star_arr, eta_arr, e_dmr_all_icl, e_star_all_icl, h_dmr_all_icl, h_star_all_icl, f_strip_dmr_icl, f_strip_star_icl)

clip_mins = clip_maxs = [None] * 6

for suffix, X_data, Y_data in [('all', X, Y), ('icl', X_icl, Y_icl)]:
    gpm.fit_multioutput_gp(X_data, Y_data, fname=f'grid_prediction_{suffix}.h5', n_jobs=24,
                           n_chains=24, n_tune=3000, n_draws=1000, rank_per_kernel=2,
                           clip_mins=clip_mins, clip_maxs=clip_maxs)
    os.rename('tests.pdf', f'test_{suffix}.pdf')
