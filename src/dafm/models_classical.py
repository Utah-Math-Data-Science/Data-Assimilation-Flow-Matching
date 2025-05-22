import torch
import math
import time # For timing analysis steps
from tqdm import tqdm
# import matplotlib.pyplot as plt # Uncomment for plotting GC test or RMSEs

# ##############################################################################
# # Utility Functions
# ##############################################################################

def center_ensemble(E, axis=0, rescale=False):
    """
    Centers the ensemble E along a given axis.

    Args:
        E (torch.Tensor): Ensemble tensor (e.g., N_particles x d_state).
        axis (int): Axis along which to compute the mean.
        rescale (bool): If True, rescale anomalies for unbiased covariance estimate.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Centered anomalies, ensemble mean.
    """
    x = torch.mean(E, dim=axis, keepdims=True)
    X_centered = E - x

    if rescale:
        N = E.shape[axis]
        if N > 1:
            X_centered *= torch.sqrt(torch.tensor(N / (N - 1), device=E.device, dtype=E.dtype))

    return X_centered, x

def pairwise_distances_torch(X, Y=None, domain_lengths=None):
    """
    Computes pairwise distances between rows of X and rows of Y.
    Handles periodic boundary conditions if domain_lengths are provided.

    Args:
        X (torch.Tensor): Tensor of shape (N, D_coord), N points in D_coord dimensions.
        Y (torch.Tensor, optional): Tensor of shape (M, D_coord), M points in D_coord dimensions.
                                    If None, Y = X.
        domain_lengths (torch.Tensor, optional): Tensor of shape (D_coord,) representing
                                                 the lengths of each dimension for
                                                 periodic boundary conditions.

    Returns:
        torch.Tensor: Tensor of shape (N, M) with pairwise distances.
    """
    if Y is None:
        Y = X

    X_unsqueezed = X.unsqueeze(1)  # Shape (N, 1, D_coord)
    Y_unsqueezed = Y.unsqueeze(0)  # Shape (1, M, D_coord)

    diff = X_unsqueezed - Y_unsqueezed  # Shape (N, M, D_coord)

    if domain_lengths is not None:
        domain_lengths_reshaped = domain_lengths.view(1, 1, -1) # Shape (1, 1, D_coord)
        abs_diff = torch.abs(diff)
        # For periodic domains, distance is min(abs_diff, domain_length - abs_diff)
        diff = torch.minimum(abs_diff, domain_lengths_reshaped - abs_diff)

    distances = torch.sqrt(torch.sum(diff ** 2, dim=-1)) # Shape (N, M)
    return distances

def gaspari_cohn_correlation(distances, c_radius):
    """
    Computes Gaspari-Cohn correlation coefficients.
    The function is non-zero for distances d < 2*c_radius.

    Args:
        distances (torch.Tensor): Tensor of distances.
        c_radius (float): Localization radius 'c' in Gaspari-Cohn.
                          The correlation is 0 for distances >= 2*c.

    Returns:
        torch.Tensor: Tensor of correlation coefficients.
    """
    c_radius = c_radius * 1.82  # following the DAPPER python package
    coeffs = torch.zeros_like(distances)
    # d_prime = distances / c_radius

    # Regime 1: 0 <= distances < c_radius  (i.e. 0 <= d_prime < 1)
    mask1 = distances < c_radius
    if torch.any(mask1):
        d_prime1 = distances[mask1] / c_radius
        coeffs[mask1] = ((( -0.25 * d_prime1 + 0.5) * d_prime1 + 0.625) * d_prime1 - 5.0/3.0) * d_prime1**2 + 1.0

    # Regime 2: c_radius <= distances < 2 * c_radius (i.e. 1 <= d_prime < 2)
    mask2 = (distances >= c_radius) & (distances < 2 * c_radius)
    if torch.any(mask2):
        d_prime2 = distances[mask2] / c_radius
        term_inv_dp = torch.zeros_like(d_prime2)
        valid_dp = d_prime2 > 1e-9 # Avoid division by zero if d_prime2 is extremely small (though it's >= 1 here)
        if torch.any(valid_dp): # Ensure there are valid entries before division
            term_inv_dp[valid_dp] = - (2.0/3.0) / d_prime2[valid_dp]

        coeffs[mask2] = ((((1.0/12.0 * d_prime2 - 0.5) * d_prime2 + 0.625) * d_prime2 + 5.0/3.0) * d_prime2 - 5.0) * d_prime2 + 4.0 + term_inv_dp

    coeffs = torch.clamp(coeffs, min=0.0, max=1.0) # Ensure valid correlation
    return coeffs


def apply_inflation(ensemble, inflation_factor):
    """
    Applies multiplicative inflation to the ensemble anomalies.

    Args:
        ensemble (torch.Tensor): Ensemble tensor (N_particles x d_state).
        inflation_factor (float): Multiplicative inflation factor.

    Returns:
        torch.Tensor: Inflated ensemble.
    """
    if inflation_factor is None or inflation_factor == 1.0:
        return ensemble

    anomalies, mean_ens = center_ensemble(ensemble, axis=0, rescale=False)
    inflated_ensemble = mean_ens + inflation_factor * anomalies
    return inflated_ensemble

def matrix_sqrt_psd(A, tol=1e-9):
    """
    Compute the square root of a symmetric positive semi-definite matrix A = V S V^T.
    Returns V S^(1/2) V^T.

    Args:
        A (torch.Tensor): Symmetric PSD matrix (..., N, N).
        tol (float): Tolerance for eigenvalue clamping.

    Returns:
        torch.Tensor: Matrix square root (..., N, N).
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(A) # For symmetric matrices
    eigenvalues_sqrt = torch.sqrt(torch.clamp(eigenvalues, min=tol)) # Clamp to avoid NaN for small negatives

    if A.ndim > 2: # Batched
        return eigenvectors @ torch.diag_embed(eigenvalues_sqrt) @ eigenvectors.transpose(-2, -1)
    else: # Single matrix
        return eigenvectors @ torch.diag(eigenvalues_sqrt) @ eigenvectors.T


# ##############################################################################
# # Bootstrap Particle Filter (Analysis Step Only)
# ##############################################################################

def bootstrap_particle_filter_analysis(
    particles_forecast, # Input is now the forecast particles
    observation_y,
    observation_operator, # Takes single particle (d_state,) -> (d_obs,)
    sigma_y,
    resampling_method="systematic"
):
    """
    Performs the analysis (update and resampling) step of the Bootstrap Particle Filter.

    Args:
        particles_forecast (torch.Tensor): Forecasted particles (N_particles x d_state).
        observation_y (torch.Tensor): Observation at current time t (d_obs,).
        observation_operator (callable): Observation mapping: y_pred = h(x_forecast).
                                        Takes (d_state,) -> (d_obs,).
        sigma_y (float): Standard deviation of observation noise (isotropic Gaussian).
        resampling_method (str): "multinomial" or "systematic".

    Returns:
        torch.Tensor: Analysis particles at current time t (N_particles x d_state).
    """
    N_particles, d_state = particles_forecast.shape
    device = particles_forecast.device
    dtype = particles_forecast.dtype

    # 1. Update (Compute Weights based on forecast particles)
    log_weights = torch.zeros(N_particles, device=device, dtype=dtype)
    if observation_y is not None:
        for i in range(N_particles):
            y_forecast_i = observation_operator(particles_forecast[i])
            log_likelihood_i = -0.5 * torch.sum(((observation_y - y_forecast_i) / sigma_y) ** 2)
            log_weights[i] = log_likelihood_i

        max_log_weight = torch.max(log_weights)
        weights = torch.exp(log_weights - max_log_weight)
        weights_sum = torch.sum(weights)
        if weights_sum > 1e-9: # Avoid division by zero if all weights are tiny
            weights /= weights_sum
        else: # Degenerate case, assign uniform weights
            weights = torch.ones(N_particles, device=device, dtype=dtype) / N_particles
    else:
        weights = torch.ones(N_particles, device=device, dtype=dtype) / N_particles

    # 2. Resampling
    weights = weights.cpu()
    if resampling_method == "multinomial":
        indices = torch.multinomial(weights, N_particles, replacement=True)
    elif resampling_method == "systematic":
        cdf = torch.cumsum(weights, dim=0)
        cdf[-1] = 1.0
        u_start = torch.rand(1, dtype=dtype) / N_particles
        u_samples = u_start + torch.arange(N_particles, dtype=dtype) / N_particles
        indices = torch.searchsorted(cdf, u_samples)
    else:
        raise ValueError(f"Unknown resampling method: {resampling_method}")
    indices = indices.to(device)

    particles_analysis = particles_forecast[indices]
    return particles_analysis


# ##############################################################################
# # Ensemble Kalman Filters (EnKF) (Analysis Step Only)
# ##############################################################################

def _enkf_pert_obs_analysis(
    ensemble_f, # Input is forecast ensemble
    observation_y,
    observation_operator_ens, # Takes (N, d_state) -> (N, d_obs)
    sigma_y,
    localization_matrix_Lxy=None, # d_state x d_obs
    localization_matrix_Lyy=None  # d_obs x d_obs
):
    """ EnKF with Perturbed Observations - Analysis Step """
    N_ensemble, d_state = ensemble_f.shape
    d_obs = observation_y.shape[0]
    device = ensemble_f.device
    dtype = ensemble_f.dtype

    ensemble_y_f = observation_operator_ens(ensemble_f)

    Af, mean_f = center_ensemble(ensemble_f, axis=0, rescale=False)
    AYf, mean_yf = center_ensemble(ensemble_y_f, axis=0, rescale=False)

    scaling_factor = 1.0 / (N_ensemble - 1) if N_ensemble > 1 else 1.0

    Pxy = (Af.T @ AYf) * scaling_factor
    Pyy = (AYf.T @ AYf) * scaling_factor

    if localization_matrix_Lxy is not None:
        Pxy = Pxy * localization_matrix_Lxy
    if localization_matrix_Lyy is not None:
        Pyy = Pyy * localization_matrix_Lyy

    R_obs = (sigma_y**2) * torch.eye(d_obs, device=device, dtype=dtype)
    innovation_cov = Pyy + R_obs

    try:
        kalman_gain = torch.linalg.solve(innovation_cov.T, Pxy.T).T
    except torch.linalg.LinAlgError:
        kalman_gain = Pxy @ torch.linalg.pinv(innovation_cov)

    obs_perturbations = sigma_y * torch.randn(N_ensemble, d_obs, device=device, dtype=dtype)
    perturbed_obs = observation_y.unsqueeze(0) + obs_perturbations
    innovations = perturbed_obs - ensemble_y_f
    ensemble_a = ensemble_f + innovations @ kalman_gain.T

    return ensemble_a, kalman_gain


def _ersf_analysis( # Ensemble Randomized Square Root Filter (ETKF variant)
    ensemble_f, # Input is forecast ensemble
    observation_y,
    observation_operator_ens, # Takes (N, d_state) -> (N, d_obs)
    sigma_y
):
    """ Ensemble Randomized Square Root Filter (ETKF) - Analysis Step """
    N_ensemble, d_state = ensemble_f.shape
    device = ensemble_f.device
    dtype = ensemble_f.dtype

    N1 = N_ensemble - 1.0; N1 = max(N1, 1.0)

    ensemble_y_f = observation_operator_ens(ensemble_f)

    Af, mean_f = center_ensemble(ensemble_f, axis=0, rescale=False)
    AYf, mean_yf = center_ensemble(ensemble_y_f, axis=0, rescale=False)

    C_tilde_sym = (AYf @ AYf.T) / (sigma_y**2) + N1 * torch.eye(N_ensemble, device=device, dtype=dtype)
    eig_vals, eig_vecs = torch.linalg.eigh(C_tilde_sym)
    eig_vals_clamped = torch.clamp(eig_vals, min=1e-9)

    T_transform_matrix = eig_vecs @ torch.diag_embed(eig_vals_clamped**-0.5) @ eig_vecs.T * torch.sqrt(torch.tensor(N1, device=device, dtype=dtype))
    Pw_term = eig_vecs @ torch.diag_embed(eig_vals_clamped**-1) @ eig_vecs.T

    innovation_dy = (observation_y.unsqueeze(0) - mean_yf)
    w_gain_transpose = (innovation_dy @ AYf.T) @ Pw_term / (sigma_y**2)

    mean_a = mean_f + w_gain_transpose @ Af
    Af_updated = T_transform_matrix @ Af
    ensemble_a = mean_a + Af_updated

    return ensemble_a, None

def _letkf_core_etkf_update(
    local_E_f_mean,      # Mean of local state forecast ensemble (N_x_local,)
    local_A_f,           # Anomalies of local state forecast ensemble (N_ens x N_x_local)
    eff_AY_f_anom,       # Effective (transformed, localized) obs anomalies (N_ens x N_y_local)
    eff_d_f_innov,       # Effective (transformed, localized) innovation (N_y_local,)
    N_ensemble
):
    """
    Core ETKF update for a local patch, assuming R_eff = I.
    This is called by _letkf_analysis.
    """
    device = local_A_f.device; dtype = local_A_f.dtype
    N1 = N_ensemble - 1.0; N1 = max(N1, 1.0)

    Pa_tilde_inv_sqrt = eff_AY_f_anom @ eff_AY_f_anom.T + \
                        N1 * torch.eye(N_ensemble, device=device, dtype=dtype)

    eig_vals, eig_vecs = torch.linalg.eigh(Pa_tilde_inv_sqrt)
    eig_vals_clamped = torch.clamp(eig_vals, min=1e-9)

    T_transform = eig_vecs @ torch.diag_embed(eig_vals_clamped**-0.5) @ eig_vecs.T * \
                  torch.sqrt(torch.tensor(N1, device=device, dtype=dtype))
    Pw = eig_vecs @ torch.diag_embed(eig_vals_clamped**-1) @ eig_vecs.T
    w_gain_transpose = (eff_d_f_innov.unsqueeze(0) @ eff_AY_f_anom.T) @ Pw
    local_mean_a = local_E_f_mean.unsqueeze(0) + w_gain_transpose @ local_A_f
    local_A_a = T_transform @ local_A_f

    return local_mean_a.squeeze(0), local_A_a


def _letkf_analysis(
    ensemble_f,
    observation_y,
    observation_operator_ens,
    sigma_y, # Scalar observation error standard deviation
    localization_radius,
    coords_state,
    coords_obs,
    domain_lengths=None
):
    """ Local Ensemble Transform Kalman Filter (LETKF) - Analysis Step """
    N_ensemble, d_state = ensemble_f.shape
    d_obs = observation_y.shape[0]
    device = ensemble_f.device; dtype = ensemble_f.dtype

    ensemble_y_f = observation_operator_ens(ensemble_f)
    Af_global, mean_f_global = center_ensemble(ensemble_f, axis=0, rescale=False)
    AYf_global, mean_yf_global = center_ensemble(ensemble_y_f, axis=0, rescale=False)
    innovation_mean_global = observation_y.unsqueeze(0) - mean_yf_global
    AYf_global_transformed = AYf_global / sigma_y
    innovation_mean_global_transformed = innovation_mean_global / sigma_y

    ensemble_a_mean_parts = torch.zeros_like(mean_f_global)
    ensemble_a_anom_parts = torch.zeros_like(Af_global)

    for k_state_idx in range(d_state):
        current_mean_f_k_scalar = mean_f_global[0, k_state_idx]
        current_Af_k_column = Af_global[:, k_state_idx]

        dist_state_k_to_obs = pairwise_distances_torch(
            coords_state[k_state_idx].unsqueeze(0), coords_obs, domain_lengths=domain_lengths).squeeze(0)
        rho_k = gaspari_cohn_correlation(dist_state_k_to_obs, localization_radius)
        local_obs_indices = torch.where(rho_k > 1e-6)[0]

        if len(local_obs_indices) == 0:
            ensemble_a_mean_parts[0, k_state_idx] = current_mean_f_k_scalar
            ensemble_a_anom_parts[:, k_state_idx] = current_Af_k_column
            continue

        AYf_local_k_transformed = AYf_global_transformed[:, local_obs_indices]
        innov_local_k_transformed = innovation_mean_global_transformed[:, local_obs_indices]
        rho_local_k = rho_k[local_obs_indices]
        sqrt_rho_local_k = torch.sqrt(rho_local_k).unsqueeze(0)
        eff_AYf_k_anom = AYf_local_k_transformed * sqrt_rho_local_k
        eff_innov_k = innov_local_k_transformed * sqrt_rho_local_k
        final_eff_AYf_k_anom = eff_AYf_k_anom
        final_eff_innov_k = eff_innov_k.squeeze(0)

        updated_mean_k, updated_A_k = _letkf_core_etkf_update(
            current_mean_f_k_scalar, current_Af_k_column.unsqueeze(-1),
            final_eff_AYf_k_anom, final_eff_innov_k, N_ensemble )

        ensemble_a_mean_parts[0, k_state_idx] = updated_mean_k
        ensemble_a_anom_parts[:, k_state_idx] = updated_A_k.squeeze(-1)

    ensemble_a = ensemble_a_mean_parts + ensemble_a_anom_parts
    return ensemble_a, None


def ensemble_kalman_filter_analysis(
    ensemble_f, observation_y, observation_operator_ens, sigma_y,
    method="EnKF-PertObs", do_inflation=True, inflation_factor=1.0,
    localization_matrix_Lxy=None, localization_matrix_Lyy=None,
    localization_radius_letkf=None, coords_state_letkf=None,
    coords_obs_letkf=None, domain_lengths_letkf=None ):
    kalman_gain_or_transform = None; ensemble_a_raw = None
    if observation_y is None:
        ensemble_a_raw = ensemble_f
    elif method == "EnKF-PertObs":
        ensemble_a_raw, kalman_gain_or_transform = _enkf_pert_obs_analysis(
            ensemble_f, observation_y, observation_operator_ens, sigma_y,
            localization_matrix_Lxy, localization_matrix_Lyy )
    elif method == "ERSF":
        ensemble_a_raw, kalman_gain_or_transform = _ersf_analysis(
            ensemble_f, observation_y, observation_operator_ens, sigma_y )
    elif method == "LETKF":
        if localization_radius_letkf is None or coords_state_letkf is None or coords_obs_letkf is None:
            raise ValueError("LETKF requires localization_radius, coords_state, and coords_obs.")
        ensemble_a_raw, kalman_gain_or_transform = _letkf_analysis(
            ensemble_f, observation_y, observation_operator_ens, sigma_y,
            localization_radius_letkf, coords_state_letkf, coords_obs_letkf, domain_lengths_letkf )
    else: raise ValueError(f"Unknown EnKF method: {method}")
    if do_inflation:
        ensemble_analysis = apply_inflation(ensemble_a_raw, inflation_factor)
    else:
        ensemble_analysis = ensemble_a_raw
    return ensemble_analysis, kalman_gain_or_transform

# ##############################################################################
# # Lorenz 96 and RK4 for Testing
# ##############################################################################
def lorenz96_rhs(x, F):
    D = x.shape[0]; dxdt = torch.zeros_like(x)
    for i in range(D):
        dxdt[i] = (x[(i + 1) % D] - x[(i - 2 + D) % D]) * x[(i - 1 + D) % D] - x[i] + F
    return dxdt

def rk4_step(rhs_func, x, dt, F_l96_param): # Renamed F_L96 to avoid conflict
    """Performs one RK4 step for a given RHS function."""
    k1 = rhs_func(x, F_l96_param)
    k2 = rhs_func(x + 0.5 * dt * k1, F_l96_param)
    k3 = rhs_func(x + 0.5 * dt * k2, F_l96_param)
    k4 = rhs_func(x + dt * k3, F_l96_param)
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next

# ##############################################################################
# # Main Test Script
# ##############################################################################
if __name__ == '__main__':
    print("Running Lorenz 96 DA example with revised LETKF and analysis timing...")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cuda' # User specified
    dtype = torch.float32
    print(f"Using device: {device}, dtype: {dtype}")

    D_L96 = 100; F_L96_const = 8.0; dt_rk4 = 0.01 # Renamed F_L96 to F_L96_const
    num_rk4_steps_between_analyses = 10; dt_analysis = dt_rk4 * num_rk4_steps_between_analyses
    N_ensemble = 20; sigma_d_val = 0; sigma_y_val = 0.05; inflation_val = 1.1

    d_obs_l96 = D_L96
    # H_matrix_l96 = torch.eye(d_obs_l96, D_L96, device=device, dtype=dtype)

    def obs_op_l96_particle(x_curr): return torch.arctan(x_curr)
    def obs_op_l96_ensemble(ensemble_curr): return torch.arctan(ensemble_curr)

    true_state_t = torch.rand(D_L96, device=device, dtype=dtype) * 3.
    # print("Spinning up true state...")
    for _ in tqdm(range(100), desc="Spinning up true state..."): # Spin-up iterations
        for _ in range(num_rk4_steps_between_analyses): # RK4 steps per assimilation cycle for spinup
            true_state_t = rk4_step(lorenz96_rhs, true_state_t, dt_rk4, F_L96_const)
    print("True state spin-up complete.")

    # initial_ensemble = true_state_t.unsqueeze(0) + torch.randn(N_ensemble, D_L96, device=device, dtype=dtype) * 0.5
    initial_ensemble = torch.randn(N_ensemble, D_L96, device=device, dtype=dtype) * 1.
    current_particles_bpf = initial_ensemble.clone()
    current_ensemble_enkf_po = initial_ensemble.clone()
    current_ensemble_ersf = initial_ensemble.clone()
    current_ensemble_letkf = initial_ensemble.clone()

    coords_state_l96 = torch.arange(D_L96, device=device, dtype=dtype).unsqueeze(1)
    coords_obs_l96 = torch.arange(d_obs_l96, device=device, dtype=dtype).unsqueeze(1)
    domain_lengths_l96 = torch.tensor([D_L96], device=device, dtype=dtype)
    loc_radius_gc = 3.
    Lxy_l96 = gaspari_cohn_correlation(
        pairwise_distances_torch(coords_state_l96, coords_obs_l96, domain_lengths=domain_lengths_l96), loc_radius_gc)
    Lyy_l96 = gaspari_cohn_correlation(
        pairwise_distances_torch(coords_obs_l96, coords_obs_l96, domain_lengths=domain_lengths_l96), loc_radius_gc)

    num_analysis_cycles = 50
    print(f"\nRunning {num_analysis_cycles} analysis cycles (dt_analysis={dt_analysis:.2f})...")
    results_rmse = {"BPF": [], "EnKF-PO": [], "ERSF": [], "LETKF": []}
    analysis_times = {"BPF": [], "EnKF-PO": [], "ERSF": [], "LETKF": []} # For storing analysis times

    for cycle in tqdm(range(num_analysis_cycles), desc='Analysis'):
        # 1. Forecast Period (True State)
        for _ in range(num_rk4_steps_between_analyses):
            true_state_t = rk4_step(lorenz96_rhs, true_state_t, dt_rk4, F_L96_const)

        # 2. Generate Observation from true state
        observation_y_t = obs_op_l96_particle(true_state_t) + \
                          sigma_y_val * torch.randn(d_obs_l96, device=device, dtype=dtype)

        # 3. Forecast Period (Ensembles/Particles) + Add Model Error
        ensembles_to_forecast = {
            # "BPF": current_particles_bpf,
            # "EnKF-PO": current_ensemble_enkf_po,
            # "ERSF": current_ensemble_ersf,
            "LETKF": current_ensemble_letkf
        }
        forecast_inputs = {}

        for name, ens in ensembles_to_forecast.items():
            forecasted_ens = torch.zeros_like(ens)
            for i in range(N_ensemble):
                member_trajectory = ens[i].clone()
                for _ in range(num_rk4_steps_between_analyses):
                    member_trajectory = rk4_step(lorenz96_rhs, member_trajectory, dt_rk4, F_L96_const)
                forecasted_ens[i] = member_trajectory
            forecasted_ens += sigma_d_val * torch.randn_like(forecasted_ens)
            forecast_inputs[name] = forecasted_ens

        # 4. Analysis Step for each method (with timing)
        # BPF
        # start_time = time.perf_counter()
        # current_particles_bpf = bootstrap_particle_filter_analysis(
        #     forecast_inputs["BPF"], observation_y_t,
        #     obs_op_l96_particle,
        #     sigma_y_val
        # )
        # end_time = time.perf_counter()
        # analysis_times["BPF"].append(end_time - start_time)
        # results_rmse["BPF"].append(torch.sqrt(torch.mean((current_particles_bpf.mean(dim=0) - true_state_t)**2)).item())

        # EnKF-PertObs
        # start_time = time.perf_counter()
        # current_ensemble_enkf_po, _ = ensemble_kalman_filter_analysis(
        #     forecast_inputs["EnKF-PO"], observation_y_t,
        #     obs_op_l96_ensemble,
        #     sigma_y_val,
        #     method="EnKF-PertObs",
        #     inflation_factor=inflation_val,
        #     localization_matrix_Lxy=Lxy_l96,
        #     localization_matrix_Lyy=Lyy_l96
        # )
        # end_time = time.perf_counter()
        # analysis_times["EnKF-PO"].append(end_time - start_time)
        # results_rmse["EnKF-PO"].append(torch.sqrt(torch.mean((current_ensemble_enkf_po.mean(dim=0) - true_state_t)**2)).item())

        # ERSF
        # start_time = time.perf_counter()
        # current_ensemble_ersf, _ = ensemble_kalman_filter_analysis(
        #     forecast_inputs["ERSF"], observation_y_t,
        #     obs_op_l96_ensemble,
        #     sigma_y_val,
        #     method="ERSF",
        #     inflation_factor=inflation_val
        # )
        # end_time = time.perf_counter()
        # analysis_times["ERSF"].append(end_time - start_time)
        # results_rmse["ERSF"].append(torch.sqrt(torch.mean((current_ensemble_ersf.mean(dim=0) - true_state_t)**2)).item())

        # LETKF
        start_time = time.perf_counter()
        current_ensemble_letkf, _ = ensemble_kalman_filter_analysis(
            forecast_inputs["LETKF"], observation_y_t,
            obs_op_l96_ensemble,
            sigma_y_val,
            method="LETKF",
            inflation_factor=inflation_val,
            localization_radius_letkf=loc_radius_gc,
            coords_state_letkf=coords_state_l96,
            coords_obs_letkf=coords_obs_l96,
            domain_lengths_letkf=domain_lengths_l96
        )
        end_time = time.perf_counter()
        analysis_times["LETKF"].append(end_time - start_time)
        results_rmse["LETKF"].append(torch.sqrt(torch.mean((current_ensemble_letkf.mean(dim=0) - true_state_t)**2)).item())

        if (cycle + 1) % 1 == 0 or cycle == num_analysis_cycles - 1:
            print(f"Cycle {cycle + 1:3d}: RMSEs -> "
                  # + f"BPF: {results_rmse['BPF'][-1]:.3f}; Time: {analysis_times['BPF'][-1]:.3f}, "
                  # + f"EnKF-PO: {results_rmse['EnKF-PO'][-1]:.3f}; Time: {analysis_times['EnKF-PO'][-1]:.3f}, "
                  # + f"ERSF: {results_rmse['ERSF'][-1]:.3f}; Time: {analysis_times['ERSF'][-1]:.3f}, "
                  + f"LETKF: {results_rmse['LETKF'][-1]:.3f}; Time: {analysis_times['LETKF'][-1]:.3f}."
            )

    print("\nExample L96 DA run completed.")

    # Calculate and print average analysis times
    print("\n--- Average Analysis Step Times ---")
    for method_name, times_list in analysis_times.items():
        if times_list:
            avg_time = sum(times_list) / len(times_list)
            print(f"{method_name}: {avg_time:.6f} seconds per analysis step")
        else:
            print(f"{method_name}: No analysis steps timed.")

    # --- Optional: Plot RMSEs ---
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 7))
    # for method_name, rmse_list in results_rmse.items():
    #     plt.plot(range(1, num_analysis_cycles + 1), rmse_list, label=method_name, marker='o', linestyle='--')
    # plt.xlabel("Analysis Cycle")
    # plt.ylabel("RMSE (vs True State)")
    # plt.title(f"DA Method Comparison on Lorenz 96 (D={D_L96}, N_ens={N_ensemble})")
    # plt.legend()
    # plt.grid(True)
    # plt.yscale('log')
    # plt.tight_layout()
    # plt.show()
