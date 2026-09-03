from imports import *


def normcdf(x, mu=0.0, sigma=1.0):
    return (1.0 + erf((x - mu) / (sigma * np.sqrt(2.0)))) / 2.0

def normpdf(x, mu=0.0, sigma=1.0):
    return (np.exp(-((x - mu) / sigma) ** 2 / 2)) / (sigma * np.sqrt(2 * np.pi))


def sigma_from_latent(lSigma, radius=10.5, rMin=2):
    """Log transform: sigma = exp(lSigma). Allows unbounded exploration without hard constraints.
    rMin kept for API compatibility but unused here.
    Old version: (radius - rMin) * normcdf(lSigma) + rMin
    """
    return np.exp(lSigma)

def gaussian_weight(distances, lSigma, radius=10.5, rMin=0.05):
    sigma = sigma_from_latent(lSigma, radius, rMin)
    w = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    w = w / np.sum(w)
    return w.astype(np.float32)

def beta_from_latent(lBeta):
    return np.exp(lBeta)


def compute_cf_prediction(source_time_series, distances, lSigma, radius=10.5, rMin=0.05, betaBool=False, lBeta=None):
    w = gaussian_weight(distances, lSigma, radius, rMin)
    prediction = np.dot(source_time_series, w)
    sigma = sigma_from_latent(lSigma, radius, rMin)
    if betaBool and lBeta is not None:
        beta = beta_from_latent(lBeta)
        prediction = beta * prediction
    else:
        beta = 1.0  # fitted via least squares downstream
    return {'prediction': prediction, 'sigma': sigma, 'beta': beta, 'weights': w}


def compute_log_likelihood(target_signal, source_time_series, distances, lSigma, radius=10.5, rMin=0.05, betaBool=False, lBeta=-5.0):
    pred = compute_cf_prediction(source_time_series, distances, lSigma, radius, rMin, betaBool, lBeta)

    if not betaBool:
        X = np.column_stack([pred['prediction'], np.ones(len(target_signal))])
        bHat = np.linalg.lstsq(X, target_signal, rcond=None)[0]
        beta = bHat[0]
        fitted_prediction = X @ bHat
    else:
        beta = pred['beta']
        fitted_prediction = pred['prediction']

    residuals = target_signal - fitted_prediction
    var_residuals = np.var(residuals)
    var_target = np.var(target_signal)
    variance_explained = 1 - (var_residuals / var_target) if var_target > 0 else 0

    mu_resid = np.mean(residuals)
    sigma_resid = np.std(residuals)
    if sigma_resid > 0:
        log_like = np.sum(np.log(normpdf(residuals, mu_resid, sigma_resid)))
    else:
        log_like = -np.inf

    # log-normal prior on sigma centered at mid-range; soft penalty above radius
    sigma_val = pred['sigma']
    if sigma_val > radius:
        prior_sigma = (np.log(normpdf(lSigma, np.log(radius/2), 1.0))
                       - ((sigma_val - radius) ** 2) / (2 * radius ** 2))
    else:
        prior_sigma = np.log(normpdf(lSigma, np.log(radius/2), 1.0))

    prior_beta = np.log(normpdf(lBeta, -2, 5)) if betaBool else 0.0
    posterior = log_like + prior_sigma + prior_beta

    return {
        'log_likelihood': log_like, 'posterior': posterior,
        'prior_sigma': prior_sigma, 'prior_beta': prior_beta,
        'sigma': pred['sigma'], 'beta': beta,
        'variance_explained': variance_explained,
        'residuals': residuals, 'var_residuals': var_residuals
    }


def propose_distance(lStepSize, maxStep):
    return np.abs(maxStep * normcdf(lStepSize) - maxStep / 2)

def propose_center_vertex(distance_proposal, distance_matrix, current_center_idx):
    distances_from_current = distance_matrix[:, current_center_idx]
    abs_diff = np.abs(distances_from_current - distance_proposal)
    candidates = np.where(abs_diff == np.min(abs_diff))[0]
    return np.random.choice(candidates)


def run_bayesian_cf_mcmc(target_signal, source_time_series, distance_matrix, n_iter=17500, radius=10.5, rMin=0.05, betaBool=False, burnIn=True, percBurnIn=10, verbose=False):
    """Metropolis-Hastings MCMC for a single target vertex."""
    n_source = source_time_series.shape[1]

    # initialise in log-space
    lSigma = np.log(radius / 2)
    lBeta = -5.0
    center_idx = np.random.randint(0, n_source)

    sigma_chain = np.zeros(n_iter)
    beta_chain = np.zeros(n_iter)
    center_chain = np.zeros(n_iter, dtype=int)
    variance_explained_chain = np.zeros(n_iter)
    log_likelihood_chain = np.zeros(n_iter)
    posterior_chain = np.zeros(n_iter)
    accepted = np.zeros(n_iter, dtype=bool)

    sigma_proposal_width = 0.1
    beta_proposal_width = 0.1
    max_step = np.max(distance_matrix) / 2

    distances_from_center = distance_matrix[:, center_idx]
    current_state = compute_log_likelihood(
        target_signal, source_time_series, distances_from_center,
        lSigma, radius, rMin, betaBool, lBeta
    )

    pbar = tqdm(range(n_iter), desc='MCMC', disable=not verbose, leave=False)
    for i in pbar:
        lSigma_prop = lSigma + np.random.normal(0, sigma_proposal_width)
        lBeta_prop = lBeta + np.random.normal(0, beta_proposal_width)

        step_proposal = np.random.normal(0, 1)
        dist_proposal = propose_distance(step_proposal, max_step)
        center_idx_prop = propose_center_vertex(dist_proposal, distance_matrix, center_idx)

        distances_from_prop_center = distance_matrix[:, center_idx_prop]
        proposed_state = compute_log_likelihood(
            target_signal, source_time_series, distances_from_prop_center,
            lSigma_prop, radius, rMin, betaBool, lBeta_prop
        )

        log_acceptance_ratio = proposed_state['posterior'] - current_state['posterior']
        if np.random.rand() < np.exp(min(0, log_acceptance_ratio)):
            lSigma = lSigma_prop
            lBeta = lBeta_prop
            center_idx = center_idx_prop
            current_state = proposed_state
            accepted[i] = True

        sigma_chain[i] = current_state['sigma']
        beta_chain[i] = current_state['beta']
        center_chain[i] = center_idx
        variance_explained_chain[i] = current_state['variance_explained']
        log_likelihood_chain[i] = current_state['log_likelihood']
        posterior_chain[i] = current_state['posterior']

        if (i + 1) % 100 == 0:
            pbar.set_postfix({'acc_rate': f'{np.mean(accepted[:i+1]):.3f}'})
    pbar.close()

    if burnIn and n_iter > percBurnIn:
        burn_idx = int(np.ceil(n_iter * percBurnIn / 100))
        sigma_chain = sigma_chain[burn_idx:]
        beta_chain = beta_chain[burn_idx:]
        center_chain = center_chain[burn_idx:]
        variance_explained_chain = variance_explained_chain[burn_idx:]
        log_likelihood_chain = log_likelihood_chain[burn_idx:]
        posterior_chain = posterior_chain[burn_idx:]
        accepted = accepted[burn_idx:]

    best_idx = np.argmax(log_likelihood_chain)
    return {
        'best_sigma': sigma_chain[best_idx],
        'best_beta': beta_chain[best_idx],
        'best_center_idx': center_chain[best_idx],
        'best_variance_explained': variance_explained_chain[best_idx],
        'best_log_likelihood': log_likelihood_chain[best_idx],
        'best_posterior': posterior_chain[best_idx],
        'sigma_chain': sigma_chain,
        'beta_chain': beta_chain,
        'center_idx_chain': center_chain,
        'variance_explained_chain': variance_explained_chain,
        'log_likelihood_chain': log_likelihood_chain,
        'posterior_chain': posterior_chain,
        'acceptance_rate': np.mean(accepted),
        'n_iterations': len(sigma_chain)
    }


def run_bayesian_cf_numpy(source_data, target_data, distance_matrix, target_indices, target_roi_labels, n_iter=17500, radius=10.5, rMin=0.05, betaBool=False, burnIn=True, percBurnIn=10, n_jobs=8, verbose=True):
    """Run MCMC for all target vertices. Returns a DataFrame with one row per vertex."""
    import time
    from tqdm import tqdm

    start_time = time.time()
    results_list = []

    for i in tqdm(range(len(target_indices)), desc="Processing vertices"):
        results_list.append(
            process_single_target_mcmc_numpy(
                target_signal=target_data[i],
                source_time_series=source_data,
                distance_matrix=distance_matrix,
                target_idx=target_indices[i],
                roi_label=target_roi_labels[i],
                n_iter=n_iter, radius=radius, rMin=rMin,
                betaBool=betaBool, burnIn=burnIn, percBurnIn=percBurnIn
            )
        )

    results = [r for r in results_list if r is not None]
    results_df = pd.DataFrame(results)

    if verbose:
        n_failed = len(target_indices) - len(results_df)
        total_time = time.time() - start_time
        print(f"  done in {total_time:.1f}s ({total_time/60:.1f} min), {n_failed} failed")

    return results_df


def process_single_target_mcmc_numpy(target_signal, source_time_series, distance_matrix, target_idx, roi_label, n_iter, radius, rMin, betaBool, burnIn, percBurnIn):
    try:
        mcmc_result = run_bayesian_cf_mcmc(target_signal=target_signal, source_time_series=source_time_series, distance_matrix=distance_matrix, n_iter=n_iter, radius=radius, rMin=rMin, betaBool=betaBool, burnIn=burnIn, percBurnIn=percBurnIn, verbose=True)
        return {'target_vertex_idx': target_idx, 'target_roi_label': roi_label, 'source_vertex_idx': int(mcmc_result['best_center_idx']), 'sigma_mm': float(mcmc_result['best_sigma']), 'beta': float(mcmc_result['best_beta']), 'variance_explained': float(mcmc_result['best_variance_explained']), 'log_likelihood': float(mcmc_result['best_log_likelihood']), 'posterior': float(mcmc_result['best_posterior']), 'acceptance_rate': float(mcmc_result['acceptance_rate']), 'n_iterations': int(mcmc_result['n_iterations']), 'center_idx_chain': mcmc_result['center_idx_chain'], 'sigma_chain': mcmc_result['sigma_chain'], 'beta_chain': mcmc_result['beta_chain'], 'sigma_mean': float(np.mean(mcmc_result['sigma_chain'])), 'sigma_std': float(np.std(mcmc_result['sigma_chain'])), 'sigma_ci_lower': float(np.percentile(mcmc_result['sigma_chain'], 2.5)), 'sigma_ci_upper': float(np.percentile(mcmc_result['sigma_chain'], 97.5)), 'beta_mean': float(np.mean(mcmc_result['beta_chain'])), 'beta_std': float(np.std(mcmc_result['beta_chain'])), 'beta_ci_lower': float(np.percentile(mcmc_result['beta_chain'], 2.5)), 'beta_ci_upper': float(np.percentile(mcmc_result['beta_chain'], 97.5))}
    except Exception as e:
        print(f"  failed for target vertex {target_idx}: {e}")
        return None
