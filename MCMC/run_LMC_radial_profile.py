import numpy as np
import pickle
import gp_model_r as gpm
import os

# Set threading environment variables
print("OMP:", os.getenv("OMP_NUM_THREADS"))
print("MKL:", os.getenv("MKL_NUM_THREADS"))
print("OPENBLAS:", os.getenv("OPENBLAS_NUM_THREADS"))
print("numexpr:", os.getenv("NUMEXPR_NUM_THREADS"))

# Load pickled data
with open('./stripped_summary_rbins.pkl', 'rb') as f:
    (mtots, mdms, mstars, mratios, peris, e_ratios, am_ratios, f_stripped_stars, f_stripped_DM,
     f_stripped_stars_ICL, f_stripped_DM_ICL, mass_stars_rbins, err_stars_rbins, mass_DM_rbins,
     err_DM_rbins, e_ratios_ICL, am_ratios_ICL, e_DM_mean, e_stars_mean, am_DM_mean, am_stars_mean,
     e_DM_mean_ICL, e_stars_mean_ICL, am_DM_mean_ICL, am_stars_mean_ICL,
     e_mean_stars_rbins, e_err_stars_rbins, e_mean_DM_rbins, e_err_DM_rbins,
     am_mean_stars_rbins, am_err_stars_rbins, am_mean_DM_rbins, am_err_DM_rbins,
     vart_DM_rbins, vart_stars_rbins, vart_err_DM_rbins, vart_err_stars_rbins,
     varr_DM_rbins, varr_stars_rbins, varr_err_DM_rbins, varr_err_stars_rbins,
     r_bins, r_edges) = pickle.load(f)

# Orbital parameters
peri_orbits = np.asarray([4, 35, 90, 185, 445])
circ_orbits = np.asarray([0.11930828643367211, 0.3417820246165659, 0.5257561772547018, 0.6987225035818878, 0.9035057543970304])

# Map circularity values to pericenters
circs = [circ_orbits[peri_orbits == peri_i][0] for peri_i in peris]
pick = np.asarray(mstars) > 1
n_rbins = len(r_bins)

# Convert to numpy arrays
mass_DM_rbins = np.array(mass_DM_rbins)
mass_stars_rbins = np.array(mass_stars_rbins)
err_DM_rbins = np.array(err_DM_rbins)
err_stars_rbins = np.array(err_stars_rbins)

energy_DM_rbins = np.array(e_mean_DM_rbins)
energy_stars_rbins = np.array(e_mean_stars_rbins)
err_energy_DM_rbins = np.array(e_err_DM_rbins)
err_energy_stars_rbins = np.array(e_err_stars_rbins)

am_DM_rbins = np.array(am_mean_DM_rbins)
am_stars_rbins = np.array(am_mean_stars_rbins)
err_am_DM_rbins = np.array(am_err_DM_rbins)
err_am_stars_rbins = np.array(am_err_stars_rbins)

v_r_DM_rbins = np.array(vart_DM_rbins)
v_r_stars_rbins = np.array(vart_stars_rbins)
err_v_r_DM_rbins = np.array(vart_err_DM_rbins)
err_v_r_stars_rbins = np.array(vart_err_stars_rbins)

v_t_DM_rbins = np.array(varr_DM_rbins)
v_t_stars_rbins = np.array(varr_stars_rbins)
err_v_t_DM_rbins = np.array(varr_err_DM_rbins)
err_v_t_stars_rbins = np.array(varr_err_stars_rbins)

# Select data
M_star_selected = np.array(mstars, dtype=np.float64)[pick]
eta_selected = np.array(circs, dtype=np.float64)[pick]
n_combinations = len(M_star_selected)

# Prepare features and targets
M_star_tile = np.repeat(M_star_selected, n_rbins)
eta_tile = np.repeat(eta_selected, n_rbins)
r_tile = np.tile(r_bins, n_combinations)

y_list = [
    np.log10(mass_DM_rbins[pick, :].flatten()),
    np.log10(mass_stars_rbins[pick, :].flatten()),
    np.log10(energy_DM_rbins[pick, :].flatten()),
    np.log10(energy_stars_rbins[pick, :].flatten()),
    np.log10(am_DM_rbins[pick, :].flatten()),
    np.log10(am_stars_rbins[pick, :].flatten()),
    np.log10(v_r_DM_rbins[pick, :].flatten()),
    np.log10(v_r_stars_rbins[pick, :].flatten()),
    np.log10(v_t_DM_rbins[pick, :].flatten()),
    np.log10(v_t_stars_rbins[pick, :].flatten())
]

y_err_list = [
    0.434 * err_DM_rbins[pick, :].flatten() / mass_DM_rbins[pick, :].flatten(),
    0.434 * err_stars_rbins[pick, :].flatten() / mass_stars_rbins[pick, :].flatten(),
    0.434 * err_energy_DM_rbins[pick, :].flatten() / energy_DM_rbins[pick, :].flatten(),
    0.434 * err_energy_stars_rbins[pick, :].flatten() / energy_stars_rbins[pick, :].flatten(),
    0.434 * err_am_DM_rbins[pick, :].flatten() / am_DM_rbins[pick, :].flatten(),
    0.434 * err_am_stars_rbins[pick, :].flatten() / am_stars_rbins[pick, :].flatten(),
    0.434 * err_v_r_DM_rbins[pick, :].flatten() / v_r_DM_rbins[pick, :].flatten(),
    0.434 * err_v_r_stars_rbins[pick, :].flatten() / v_r_stars_rbins[pick, :].flatten(),
    0.434 * err_v_t_DM_rbins[pick, :].flatten() / v_t_DM_rbins[pick, :].flatten(),
    0.434 * err_v_t_stars_rbins[pick, :].flatten() / v_t_stars_rbins[pick, :].flatten()
]

# Align data lengths
min_len = min(len(M_star_tile), len(y_list[0]), len(eta_tile), len(r_tile))
X = np.column_stack([np.log10(M_star_tile[:min_len]), eta_tile[:min_len], np.log10(r_tile[:min_len])])
Y = np.column_stack([y[:min_len] for y in y_list])

mask_valid = np.all(np.isfinite(Y), axis=1)
X = X[mask_valid]
Y = Y[mask_valid]
Y_err = np.column_stack([y_err[:min_len][mask_valid] for y_err in y_err_list])

# Fit LMC models
n_tune = 3000
n_draws = 500

out_all = gpm.fit_multioutput_gp(X, Y[:, [0, 1]], Y_err=Y_err[:, [0, 1]], 
                                 fname='grid_prediction_r.h5', n_jobs=24,
                                 n_chains=24, n_tune=n_tune, n_draws=n_draws, rank_per_kernel=1)
os.rename('tests_r.pdf', 'test_r_r.pdf')
