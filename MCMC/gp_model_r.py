import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from scipy.linalg import cho_factor, cho_solve

import pymc as pm
from pymc.variational.callbacks import CheckParametersConvergence
import arviz as az
import pytensor.tensor as pyt


class MultiOutputGPModel:
    """Multi-output GP model with MCMC fitting, posterior checks, and grid realizations."""

    def __init__(self, kernel_constructors=None, rank_per_kernel=2, random_seed=1):
        self.kernel_constructors = kernel_constructors or [pm.gp.cov.ExpQuad, pm.gp.cov.Matern52]
        self.rank_per_kernel = rank_per_kernel
        self.random_seed = random_seed

    @staticmethod
    def rbf_kernel(X1, X2, ls):
        X1s, X2s = X1 / ls, X2 / ls
        sqd = np.sum((X1s[:, None, :] - X2s[None, :, :])**2, axis=2)
        return np.exp(-0.5 * sqd)

    @staticmethod
    def matern52_kernel(X1, X2, ls):
        X1s, X2s = X1 / ls, X2 / ls
        d = np.sqrt(np.sum((X1s[:, None, :] - X2s[None, :, :])**2, axis=2))
        sqrt5 = np.sqrt(5.0)
        return (1.0 + sqrt5 * d + 5.0/3.0 * d**2) * np.exp(-sqrt5 * d)

    def run_MCMC(self, X_std, Y_std, Y_err=None, rank_per_kernel=None, random_seed=None,
                 n_jobs=10, n_chains=10, n_tune=10_000, n_draws=2_500,
                 target_accept=0.95):
        """Fit the PyMC marginal multi-output GP and return model objects."""
        rank_per_kernel = rank_per_kernel or self.rank_per_kernel
        random_seed = random_seed or self.random_seed

        N, input_dim = X_std.shape
        D = Y_std.shape[1]
        n_kernels = len(self.kernel_constructors)

        # Augment data for multi-output GP
        X_aug = np.vstack([np.hstack([X_std, np.full((N, 1), d)]) for d in range(D)])
        Y_aug = np.hstack([Y_std[:, d] for d in range(D)])

        if Y_err is not None:
            y_err_full = np.hstack([Y_err[:, d] for d in range(D)])
        else:
            y_err_full = 0.0
    
        full_input_dim = input_dim + 1
        print(f'Fitting model d={D}, n={N}, kernels={self.kernel_constructors}, rank per kernel={rank_per_kernel}')
        with pm.Model() as model:
            ls = pm.LogNormal("ls", mu=np.log(0.5), sigma=0.5, shape=(n_kernels, input_dim))
            if n_kernels == 2:
                amp = pm.Deterministic("amp", pyt.stack([
                    pm.HalfNormal("a0", 1.0),
                    pm.HalfNormal("a1", 0.5)
                ]))
            else:
                amp = pm.HalfNormal("amp", sigma=1.0, shape=(n_kernels,))

            cov_sum = 0
            for k, kernel_i in enumerate(self.kernel_constructors):
                cov_input = (amp[k] ** 2) * kernel_i(input_dim=full_input_dim, ls=ls[k], active_dims=list(range(input_dim)))
                if n_kernels == 1:
                    w0_pos = pm.HalfNormal("w0_pos", 0.5, shape=(rank_per_kernel,))
                    w_rest = pm.Normal("w_rest", 0.0, 0.5, shape=(D-1, rank_per_kernel))
                    W_k = pyt.vstack([w0_pos[None, :], w_rest])
                else:
                    W_k = pm.Normal(f"W_k{k}", mu=0.0, sigma=0.3, shape=(D, rank_per_kernel))
                kappa_k = pm.HalfNormal(f"kappa_k{k}", sigma=0.5, shape=(D,))
                coregion_k = pm.gp.cov.Coregion(input_dim=full_input_dim, W=W_k, kappa=kappa_k, active_dims=[input_dim])
                cov_sum += cov_input * coregion_k

            #sigma = pm.HalfNormal("sigma", sigma=0.075)
            sigma_jitter = 1e-5
            mogp = pm.gp.Marginal(cov_func=cov_sum)
            mogp.marginal_likelihood("y", X=X_aug, y=Y_aug,
                                    sigma=pyt.sqrt(y_err_full**2 + sigma_jitter**2))

            # ADVI + jittered starts
            advi_iters = 10_000
            advi_callback = CheckParametersConvergence(tolerance=1e-3, diff="absolute")
            approx = pm.fit(advi_iters, method="advi",
                            obj_optimizer=pm.adam(learning_rate=1e-2),
                            callbacks=[advi_callback],
                            random_seed=random_seed)
            advi_idata = approx.sample(1000)

            def jitter_advi_from_idata(idata, scale=0.1, min_pos=1e-9):
                start = {}
                for var in idata.posterior.data_vars:
                    vals = idata.posterior[var].values
                    mean = vals.mean(axis=(0, 1))
                    std = vals.std(axis=(0, 1))
                    jitter_scale = scale * (std + np.abs(mean) + 1e-12)
                    jitter = np.random.normal(loc=0.0, scale=jitter_scale, size=mean.shape)
                    v = mean + jitter
                    if np.all(vals > 0):
                        v = np.maximum(v, min_pos)
                    start[var] = v
                return start

            init_list = [jitter_advi_from_idata(advi_idata, scale=0.025) for _ in range(n_chains)]
            n_jobs_effective = min(n_chains, n_jobs, (os.cpu_count() or 1))
            idata = pm.sample(
                tune=n_tune,
                draws=n_draws,
                chains=n_chains,
                cores=n_jobs_effective,
                initvals=init_list,
                target_accept=target_accept,
                max_treedepth=10,
                return_inferencedata=True,
                random_seed=random_seed
            )
        return model, mogp, idata, X_aug, Y_aug
    
    def evaluate_posterior_predictive(self, model, mogp, idata, X_aug, Y, scaler_Y, D, N, random_seed=None):
        """Posterior predictive checks and diagnostics for 3D input GP."""
        random_seed = random_seed or self.random_seed
        param_names = list(idata.posterior.keys())
        print("Parameter names in posterior:", param_names)
        az.plot_trace(idata, var_names=param_names, combined=False)

        # Remove old variable if present
        if "preds_train_check" in model.named_vars:
            try:
                model.free_RVs.remove(model["preds_train_check"])
            except Exception:
                pass
            del model.named_vars["preds_train_check"]

        with model:
            mogp.conditional("preds_train_check", X_aug)
            pp_train = pm.sample_posterior_predictive(
                idata, var_names=["preds_train_check"],
                random_seed=random_seed, progressbar=False
            )

        arr_t = pp_train.posterior_predictive["preds_train_check"].values
        arr_t_comb = arr_t.reshape(-1, arr_t.shape[-1])
        pred_mean_std = arr_t_comb.mean(axis=0)
        pred_std_std = arr_t_comb.std(axis=0)
        pred_mean_std_mat = pred_mean_std.reshape(D, N).T
        pred_std_std_mat = pred_std_std.reshape(D, N).T

        pred_mean_orig = scaler_Y.inverse_transform(pred_mean_std_mat)
        pred_std_orig = pred_std_std_mat * scaler_Y.scale_

        residuals = Y - pred_mean_orig
        bias_per_output = residuals.mean(axis=0)
        std_per_output = residuals.std(axis=0)
        frac_resid = (pred_mean_orig - Y) / (np.where(np.abs(Y) > 0, Y, np.nan))
        z_scores = residuals / pred_std_orig
        z_mean = np.nanmean(z_scores, axis=0)
        z_std = np.nanstd(z_scores, axis=0)

        print("Posterior mean params:")
        print("Residual bias per output:", bias_per_output)
        print("Residual std per output:", std_per_output)
        print("Mean fractional residuals:", np.nanmean(frac_resid, axis=0))
        print("z-score mean per output (should be ~0):", z_mean)
        print("z-score std per output (should be ~1):", z_std)

        # Plotting diagnostics
        fig, axes = plt.subplots(D, 2, figsize=(12, 4 * D))
        for d in range(D):
            axes[d, 0].errorbar(
                Y[:, d], pred_mean_orig[:, d], yerr=pred_std_orig[:, d],
                marker='o', markersize=3, capsize=5,
                markerfacecolor='white', markeredgewidth=2, ls='', lw=2, color='k'
            )
            axes[d, 0].plot([Y[:, d].min(), Y[:, d].max()], [Y[:, d].min(), Y[:, d].max()],
                            ls='--', color='grey')
            axes[d, 0].set_title(f"Pred vs Data (output {d})")
            axes[d, 0].set_xlabel("Data")
            axes[d, 0].set_ylabel(r"Predicted mean $\pm \sigma$")

            axes[d, 1].hist(Y[:, d] - pred_mean_orig[:, d], bins=30, alpha=1, color='k')
            axes[d, 1].axvline(0, color='grey', linestyle='--')
            axes[d, 1].set_title(f"Residuals (data - pred), output {d}")
            axes[d, 1].set_xlabel("Residual")
            axes[d, 1].set_ylabel("Count")

        plt.tight_layout()
        fig.savefig('tests_r.pdf', bbox_inches='tight')
        
        return {
            "pred_mean_orig": pred_mean_orig,
            "pred_std_orig": pred_std_orig,
            "residuals": residuals,
            "z_scores": z_scores
        }

    def gp_grid_realizations_multi_kernel(self, idata, scaler_X, scaler_Y, X_std, Y_std,
                                        mstar_range, nu_range, r_range,
                                        n_m=50, n_nu=50, n_r=50,
                                        n_posterior_draws=300,
                                        kernel_constructors=None,
                                        jitter=1e-8,
                                        random_seed=None, n_jobs=10):
        """Generate grid realizations from the multi-kernel GP posterior in 3D input space."""
        kernel_constructors = kernel_constructors or self.kernel_constructors
        random_seed = random_seed or self.random_seed
        rng = np.random.default_rng(random_seed)

        kernel_map = {'ExpQuad': self.rbf_kernel, 'Matern52': self.matern52_kernel}
        kernel_fns = [kernel_map[k.__name__] for k in kernel_constructors]
        n_kernels = len(kernel_constructors)
        D = Y_std.shape[1]
        N = X_std.shape[0]

        # build 3D grid
        m_grid = np.linspace(*mstar_range, n_m)
        nu_grid = np.linspace(*nu_range, n_nu)
        r_grid = np.linspace(*r_range, n_r)
        grid_m, grid_nu, grid_r = np.meshgrid(m_grid, nu_grid, r_grid, indexing='ij')
        X_grid_phys = np.column_stack([grid_m.ravel(), grid_nu.ravel(), grid_r.ravel()])
        X_grid_std = scaler_X.transform(X_grid_phys)
        n_grid = X_grid_std.shape[0]

        # --- Augment training and grid for multi-output ---
        X_train_aug = np.vstack([np.hstack([X_std, np.full((N, 1), d)]) for d in range(D)])
        Y_train_aug = np.hstack([Y_std[:, d] for d in range(D)])
        out_train = X_train_aug[:, -1].astype(int)

        X_grid_aug = np.vstack([np.hstack([X_grid_std, np.full((n_grid, 1), d)]) for d in range(D)])
        out_grid = X_grid_aug[:, -1].astype(int)
        X_train_phys = np.tile(X_std, (D, 1))
        X_grid_phys_std = np.tile(X_grid_std, (D, 1))
        y_train = Y_train_aug.astype(float)

        # posterior samples
        posterior_vars = list(idata.posterior.data_vars)
        W_names = sorted([v for v in posterior_vars if v.startswith("W_k")])
        kappa_names = sorted([v for v in posterior_vars if v.startswith("kappa_k")])
        amp_var = next((v for v in posterior_vars if 'amp' in v), None)

        n_chains_total = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
        idx = rng.choice(n_chains_total, size=min(n_posterior_draws, n_chains_total), replace=False)
        chains, draws = idx // idata.posterior.sizes["draw"], idx % idata.posterior.sizes["draw"]

        pred_all = []

        def get_var(name, c, d):
            return idata.posterior[name].values[c, d]

        for c, d in tqdm(zip(chains, draws), total=len(chains)):
            Ntot, Mtot = X_train_aug.shape[0], X_grid_aug.shape[0]
            K_xx = np.zeros((Ntot, Ntot))
            K_sx = np.zeros((Mtot, Ntot))
            K_ss_diag = np.ones(Mtot)
            ls_all = np.atleast_2d(get_var("ls", c, d))

            for k in range(n_kernels):
                ls_vec = ls_all[k] if ls_all.shape[0] == n_kernels else ls_all[0]
                amp_val = float(get_var(amp_var, c, d).squeeze()[k]) if amp_var else 1.0
                kernel_fn = kernel_fns[k]

                def chunk_kernel(chunk):
                    return kernel_fn(chunk, X_train_phys, ls_vec)

                X_chunks = np.array_split(X_grid_phys_std, max(1, n_jobs))
                K_sx_chunks = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(chunk_kernel)(chunk) for chunk in X_chunks
                )
                K_sx_k = np.vstack(K_sx_chunks)
                K_phys_xx = kernel_fn(X_train_phys, X_train_phys, ls_vec)

                if W_names:
                    Wk = get_var(W_names[k], c, d).reshape(D, -1)
                    kapp = get_var(kappa_names[k], c, d).reshape(D,)
                    Bk = Wk @ Wk.T + np.diag(kapp)
                else:
                    Bk = np.eye(D)

                B_block = Bk[out_train[:, None], out_train[None, :]]
                B_block_s = Bk[out_grid[:, None], out_train[None, :]]
                B_block_ss = Bk[out_grid, out_grid]

                K_xx += (amp_val ** 2) * (K_phys_xx * B_block)
                K_sx += (amp_val ** 2) * (K_sx_k * B_block_s)
                K_ss_diag += (amp_val ** 2) * B_block_ss

            sigma_val = float(get_var("sigma", c, d)) if "sigma" in posterior_vars else 0.0
            cho = cho_factor(K_xx + np.eye(Ntot) * (sigma_val ** 2 + jitter), lower=True, check_finite=False)
            alpha = cho_solve(cho, y_train)
            m_s = K_sx @ alpha
            Kinv_Ksx = cho_solve(cho, K_sx.T)
            var_s = np.maximum(K_ss_diag - np.einsum("ij,ij->j", K_sx.T, Kinv_Ksx), 0.0) + sigma_val ** 2
            sample = m_s + np.sqrt(var_s) * rng.standard_normal(Mtot)
            pred_real = sample.reshape(D, n_m, n_nu, n_r).transpose(1, 2, 3, 0)
            pred_all.append(pred_real)

        pred_all = np.array(pred_all)
        pred_real_orig = np.array([scaler_Y.inverse_transform(pred_all[i].reshape(-1, D)).reshape(pred_all[i].shape)
                                for i in range(pred_all.shape[0])])

        return m_grid, nu_grid, r_grid, pred_real_orig


def fit_multioutput_gp(
    X, Y, Y_err=None,
    random_seed=1,
    save_hdf5=True,
    fname='gp_predictions.h5',
    n_jobs=10,
    n_chains=10,
    n_tune=10_000,
    n_draws=2_500,
    overwrite=False,
    rank_per_kernel=1
):
    """
    Fits a multi-output Gaussian Process (GP) model to 3D input data
    (m_star, nu, radius), performs MCMC sampling, evaluates posterior
    predictive checks, and generates grid-based realizations.

    Parameters
    ----------
    X : array-like, shape (N, 3)
        Input variables [m_star, nu, radius].
    Y : array-like, shape (N, D)
        Output variables.

    """

    if os.path.exists(fname) and not overwrite:
        with h5py.File(fname, 'r') as f:
            m_grid = f['m_grid'][()]
            nu_grid = f['nu_grid'][()]
            r_grid = f['r_grid'][()]
            pred_real = f['pred_realizations'][()]
        return {
            'model': None,
            'mogp': None,
            'idata': None,
            'X_aug': None,
            'Y_aug': None,
            'tests': None,
            'm_grid': m_grid,
            'nu_grid': nu_grid,
            'r_grid': r_grid,
            'pred_realizations': pred_real
        }

    scaler_X = StandardScaler().fit(X)
    X_std = scaler_X.transform(X)
    scaler_Y = StandardScaler().fit(Y)
    Y_std = scaler_Y.transform(Y)
    N, input_dim = X_std.shape
    D = Y_std.shape[1]

    print(Y_err.shape)

    print(D)

    if Y_err is not None:
        # std dev scaling: divide by the per-output standard deviation used in the scaler
        Y_err_std = Y_err / scaler_Y.scale_
    else:
        Y_err_std = None

    # visualise input at representative radii (slices in X[:,2])
    radii_sel = np.percentile(X_std[:, 2], [10, 50, 90])
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    for i, r_val in enumerate(radii_sel):
        mask = np.abs(X_std[:, 2] - r_val) < 0.05
        if np.any(mask):
            sc = axs[i].scatter(X_std[mask, 0], X_std[mask, 1],
                                c=Y_std[mask, 0], s=80, edgecolors='k')
            axs[i].set_title(f'r ≈ {r_val:.2f}')
            plt.colorbar(sc, ax=axs[i])
    plt.tight_layout()

    #  fit the multi-output GP
    gp = MultiOutputGPModel(random_seed=random_seed)
    model, mogp, idata, X_aug, Y_aug = gp.run_MCMC(
        X_std, Y_std, Y_err=Y_err_std,
        n_jobs=n_jobs, n_chains=n_chains,
        n_tune=n_tune, n_draws=n_draws, rank_per_kernel=rank_per_kernel
    )

    #  posterior predictive diagnostics
    tests = gp.evaluate_posterior_predictive(model, mogp, idata, X_aug, Y, scaler_Y, D, N)

    #  grid generation
    m_range = [X[:, 0].min() * 0.98, X[:, 0].max() * 1.0]
    nu_range = [0.0, 1.0]
    r_range = [X[:, 2].min(), X[:, 2].max()]

    print(m_range, nu_range, r_range)
    
    m_grid, nu_grid, r_grid, pred_real = gp.gp_grid_realizations_multi_kernel(
        idata, scaler_X, scaler_Y, X_std, Y_std,
        mstar_range=m_range,
        nu_range=nu_range,
        r_range=r_range,
        n_m=50, n_nu=50, n_r=20,
        n_posterior_draws=n_draws * n_chains,
        random_seed=random_seed,
        kernel_constructors=gp.kernel_constructors,
        n_jobs=n_jobs
    )

    print(m_grid)
    print(nu_grid)
    print(r_grid)

    #  save to disk
    if save_hdf5:
        with h5py.File(fname, 'w') as f:
            f.create_dataset('m_grid', data=m_grid)
            f.create_dataset('nu_grid', data=nu_grid)
            f.create_dataset('r_grid', data=r_grid)
            f.create_dataset('pred_realizations', data=pred_real)

    return {
        'model': model,
        'mogp': mogp,
        'idata': idata,
        'X_aug': X_aug,
        'Y_aug': Y_aug,
        'tests': tests,
        'm_grid': m_grid,
        'nu_grid': nu_grid,
        'r_grid': r_grid,
        'pred_realizations': pred_real
    }
