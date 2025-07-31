import torch
import math
import time # For timing analysis steps
from tqdm import tqdm
# from localization import pairwise_distances, dist2coeff
# import matplotlib.pyplot as plt # Uncomment for plotting GC test or RMSEs

# ##############################################################################
# # Utility Functions
# ##############################################################################

def center_ensemble(E, rescale=False):
    """
    Centers the ensemble E along the second dimension (dim=1).

    Args:
        E (torch.Tensor): Ensemble tensor (batch_size x N_particles x d_state).
        rescale (bool): If True, rescale anomalies for unbiased covariance estimate.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Centered anomalies, ensemble mean.
    """
    # Calculate mean along dim=1
    x = torch.mean(E, dim=1, keepdims=True)
    X_centered = E - x

    if rescale:
        # Get the size of the dimension along which mean was computed
        N = E.shape[1]
        if N > 1:
            # Rescale for unbiased covariance
            X_centered *= torch.sqrt(torch.tensor(N / (N - 1), device=E.device, dtype=E.dtype))

    return X_centered, x

def apply_inflation(ensemble, inflation_factor):
    """
    Applies multiplicative inflation to ensemble anomalies.

    Args:
        ensemble (torch.Tensor): Ensemble tensor (batch_size x N_particles x d_state).
        inflation_factor (float): Multiplicative inflation factor.

    Returns:
        torch.Tensor: Inflated ensemble (batch_size x N_particles x d_state).
    """
    if inflation_factor is None or inflation_factor == 1.0:
        return ensemble

    # center_ensemble now expects a 3D tensor and operates on dim=1
    anomalies, mean_ens = center_ensemble(ensemble, rescale=False)
    # anomalies: (B, N_particles, d_state), mean_ens: (B, 1, d_state)
    # inflation_factor is scalar, broadcasts with anomalies
    # mean_ens broadcasts correctly with (inflation_factor * anomalies)
    inflated_ensemble = mean_ens + inflation_factor * anomalies
    return inflated_ensemble

def matrix_sqrt_psd(A, tol=1e-9):
    """
    Compute the square root of symmetric positive semi-definite matrices.
    A = V S V^T. Returns V S^(1/2) V^T. Handles batched inputs.

    Args:
        A (torch.Tensor): Symmetric PSD matrix or batch of matrices (..., N, N).
        tol (float): Tolerance for eigenvalue clamping to ensure non-negativity.

    Returns:
        torch.Tensor: Matrix square root (..., N, N).
    """
    # torch.linalg.eigh handles batched inputs (e.g., batch_size x N x N)
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    # Clamp eigenvalues to be non-negative before sqrt
    eigenvalues_sqrt = torch.sqrt(torch.clamp(eigenvalues, min=tol))

    # Reconstruct the matrix square root
    # eigenvectors @ diag_matrix @ eigenvectors_transposed
    # This handles both batched (A.ndim > 2) and single (A.ndim = 2) cases correctly
    # due to how torch.diag_embed and batched matrix multiply (@) work.
    # .transpose(-2, -1) is robust for batched or non-batched.
    return eigenvectors @ torch.diag_embed(eigenvalues_sqrt) @ eigenvectors.transpose(-2, -1)



# ##############################################################################
# # Bootstrap Particle Filter (Analysis Step Only)
# ##############################################################################

from torch import vmap

def bootstrap_particle_filter_analysis(
    particles_forecast,      # (batch_size, N_particles, d_state)
    observation_y,           # (batch_size, d_obs) or (d_obs,)
    observation_operator,    # callable or torch.Tensor of shape (d_obs, d_state)
    sigma_y,                 # float (std dev of observation noise)
    resampling_method="multinomial",
    resample_on_cpu=False,   # bool: If True, moves weight/index calculations to CPU
    sigma_reg=None,          # float (std dev for regularization noise)
    use_half_precision=False,# bool: If True, use float16 to save memory
    max_chunk_size=5000      # int: Max particles to process at once for non-matrix operators
):
    """
    Performs an advanced batch analysis (update & resampling) for a Bootstrap Particle Filter.
    This version is optimized for a linear observation_operator provided as a matrix
    and includes a fix for float16 underflow during resampling.

    Args:
        particles_forecast (torch.Tensor): Forecasted particles of shape (batch_size, N_particles, d_state).
        observation_y (torch.Tensor): Observation tensor of shape (batch_size, d_obs) or (d_obs,).
        observation_operator (callable or torch.Tensor): The observation mapping.
                                 - If callable: a function y_pred = h(x_forecast).
                                 - If torch.Tensor: A matrix H of shape (d_obs, d_state).
        sigma_y (float): Standard deviation of the observation noise.
        resampling_method (str): Resampling method, either "multinomial" or "systematic".
        resample_on_cpu (bool): If True, computes weights and indices on CPU to conserve GPU memory.
        sigma_reg (float, optional): Std dev for regularization noise.
        use_half_precision (bool): If True, converts key tensors to float16.
        max_chunk_size (int): Max particles for vectorized call (only for callable operators).

    Returns:
        torch.Tensor: The analysis particles of shape (batch_size, N_particles, d_state).
    """
    batch_size, N_particles, d_state = particles_forecast.shape
    device = particles_forecast.device
    original_dtype = particles_forecast.dtype

    if N_particles == 0:
        return particles_forecast

    # 1. Precision Handling
    compute_dtype = torch.float16 if use_half_precision else original_dtype
    particles_forecast = particles_forecast.to(compute_dtype)
    if observation_y is not None:
        observation_y = observation_y.to(compute_dtype)

    # 2. Update (Compute Weights)
    if observation_y is not None:
        if observation_y.ndim == 1:
            obs_y_broadcastable = observation_y.view(1, 1, -1)
        elif observation_y.ndim == 2:
            obs_y_broadcastable = observation_y.unsqueeze(1)
        else:
            raise ValueError("observation_y must be a 1D or 2D tensor.")

        if isinstance(observation_operator, torch.Tensor):
            H = observation_operator.to(device=device, dtype=compute_dtype)
            if H.shape[1] != d_state:
                raise ValueError(f"Matrix shape mismatch: H requires {d_state} columns, but has {H.shape[1]}")
            y_forecast = particles_forecast @ H.T
            diff_sq = ((obs_y_broadcastable - y_forecast) / sigma_y) ** 2
            log_weights = -0.5 * torch.sum(diff_sq, dim=2)
        elif callable(observation_operator):
            log_weights = torch.zeros(batch_size, N_particles, device=device, dtype=compute_dtype)
            vectorized_op = vmap(vmap(observation_operator))
            for i in range(0, N_particles, max_chunk_size):
                chunk_end = min(i + max_chunk_size, N_particles)
                particles_chunk = particles_forecast[:, i:chunk_end, :]
                y_forecast_chunk = vectorized_op(particles_chunk)
                diff_sq_chunk = ((obs_y_broadcastable - y_forecast_chunk) / sigma_y) ** 2
                log_weights[:, i:chunk_end] = -0.5 * torch.sum(diff_sq_chunk, dim=2)
        else:
            raise TypeError("observation_operator must be a callable or a torch.Tensor")

        max_log_w = torch.max(log_weights, dim=1, keepdim=True)[0]
        weights_unnormalized = torch.exp(log_weights - max_log_w)
        sum_weights = torch.sum(weights_unnormalized, dim=1, keepdim=True)

        uniform_dist = torch.full((N_particles,), 1.0 / N_particles, device=device, dtype=compute_dtype)
        weights = uniform_dist.unsqueeze(0).expand(batch_size, -1).clone()

        good_batches_mask = (sum_weights > 1e-6).squeeze(-1)
        if good_batches_mask.any():
            normalized_w_good = weights_unnormalized[good_batches_mask] / sum_weights[good_batches_mask]
            weights[good_batches_mask] = normalized_w_good
    else:
        weights = torch.full((batch_size, N_particles), 1.0 / N_particles, device=device, dtype=compute_dtype)

    # 3. Resampling (FIX APPLIED HERE)
    # --- For numerical stability, cast weights to float32 for resampling calculations ---
    resample_device = torch.device("cpu") if resample_on_cpu else device
    resample_dtype = torch.float32 # Use float32 to prevent underflow

    weights_resample = weights.to(device=resample_device, dtype=resample_dtype)

    # Add a small epsilon to the weights to guarantee the sum is non-zero
    weights_resample += torch.finfo(resample_dtype).eps

    if resampling_method == "multinomial":
        indices = torch.multinomial(weights_resample, N_particles, replacement=True)
    elif resampling_method == "systematic":
        cdf = torch.cumsum(weights_resample, dim=1)
        cdf[:, -1] = 1.0
        u_start = torch.rand(batch_size, 1, device=resample_device, dtype=resample_dtype) / N_particles
        u_uniform_strata = torch.arange(N_particles, device=resample_device, dtype=resample_dtype) / N_particles
        u_samples = u_start + u_uniform_strata.unsqueeze(0)
        indices = torch.searchsorted(cdf, u_samples, right=True).clamp_(0, N_particles - 1)
    else:
        raise ValueError(f"Unknown resampling method: {resampling_method}")

    indices = indices.to(device=device)
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
    particles_analysis = particles_forecast[batch_indices, indices]

    # 4. Regularization
    if sigma_reg is not None and sigma_reg > 0:
        particles_analysis += torch.randn_like(particles_analysis) * sigma_reg

    # 5. Final Type Casting
    if use_half_precision:
        particles_analysis = particles_analysis.to(original_dtype)

    return particles_analysis


# ##############################################################################
# # Ensemble Kalman Filters (EnKF) (Analysis Step Only)
# ##############################################################################

def _enkf_pert_obs_analysis(
    ensemble_f,             # (B, N_ensemble, d_state)
    observation_y,          # (B, d_obs) or (d_obs,)
    observation_operator_ens, # (B, N_ensemble, d_state) -> (B, N_ensemble, d_obs)
    sigma_y,                # scalar or (B,)
    localization_matrix_Lxy=None, # (d_state, d_obs), broadcasts
    localization_matrix_Lyy=None  # (d_obs, d_obs), broadcasts
):
    """ EnKF with Perturbed Observations - Analysis Step (Batched) """
    batch_size, N_ensemble, d_state = ensemble_f.shape

    if observation_y.ndim == 1:
        obs_y_eff = observation_y.unsqueeze(0)
        d_obs = observation_y.shape[0]
    else:
        obs_y_eff = observation_y
        d_obs = observation_y.shape[-1]
        if obs_y_eff.shape[0] != batch_size and obs_y_eff.shape[0] != 1:
             raise ValueError("Batch size of observation_y must match ensemble_f or be 1.")

    device = ensemble_f.device
    dtype = ensemble_f.dtype

    ensemble_y_f = observation_operator_ens(ensemble_f)

    Af, _ = center_ensemble(ensemble_f, rescale=False)
    AYf, _ = center_ensemble(ensemble_y_f, rescale=False)

    scaling_factor = 1.0 / (N_ensemble - 1) if N_ensemble > 1 else 1.0

    Pxy = (Af.transpose(-2, -1) @ AYf) * scaling_factor
    Pyy = (AYf.transpose(-2, -1) @ AYf) * scaling_factor

    if localization_matrix_Lxy is not None:
        Pxy = Pxy * localization_matrix_Lxy
    if localization_matrix_Lyy is not None:
        Pyy = Pyy * localization_matrix_Lyy

    if isinstance(sigma_y, torch.Tensor) and sigma_y.ndim == 1 and sigma_y.shape[0] == batch_size:
        R_val = sigma_y.view(batch_size, 1, 1)**2
        R_obs = torch.eye(d_obs, device=device, dtype=dtype).unsqueeze(0) * R_val
    else:
        R_obs = (sigma_y**2) * torch.eye(d_obs, device=device, dtype=dtype)

    innovation_cov = Pyy + R_obs

    # --- MODIFICATION START ---
    # Add a small regularization term to innovation_cov to improve stability
    epsilon = 1e-6 # Regularization strength; adjust if necessary

    # Ensure reg_identity is correctly broadcastable for batched innovation_cov
    if innovation_cov.ndim == 3 and batch_size > 0 : # Batched
        reg_identity = torch.eye(d_obs, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
    elif innovation_cov.ndim == 2: # Non-batched (should not occur if inputs are batched)
         reg_identity = torch.eye(d_obs, device=device, dtype=dtype)
    else: # Handles batch_size = 0 or other unexpected dims for innovation_cov
        reg_identity = torch.eye(d_obs, device=device, dtype=dtype).unsqueeze(0)


    innovation_cov_reg = innovation_cov + epsilon * reg_identity
    # --- MODIFICATION END ---

    try:
        # K^T = solve(S_reg, Pxy^T) -> K = (solve(S_reg, Pxy^T))^T
        kalman_gain_T = torch.linalg.solve(innovation_cov_reg, Pxy.transpose(-2, -1))
        kalman_gain = kalman_gain_T.transpose(-2, -1)
    except torch.linalg.LinAlgError: # Catches errors like singularity if solve fails
        # Fallback to pseudo-inverse if solve fails even with regularization
        kalman_gain = Pxy @ torch.linalg.pinv(innovation_cov_reg)

    if isinstance(sigma_y, torch.Tensor) and sigma_y.ndim == 1 and sigma_y.shape[0] == batch_size:
        sigma_y_expanded = sigma_y.view(batch_size, 1, 1)
    else:
        sigma_y_expanded = sigma_y

    obs_perturbations = sigma_y_expanded * torch.randn(batch_size, N_ensemble, d_obs, device=device, dtype=dtype)
    perturbed_obs = obs_y_eff.unsqueeze(1) + obs_perturbations
    innovations = perturbed_obs - ensemble_y_f
    ensemble_a = ensemble_f + innovations @ kalman_gain.transpose(-2, -1)

    return ensemble_a, kalman_gain


def _ersf_analysis( # Ensemble Randomized Square Root Filter (ETKF variant) - Batched
    ensemble_f,             # (B, N_ensemble, d_state)
    observation_y,          # (B, d_obs) or (d_obs,)
    observation_operator_ens, # (B, N_ensemble, d_state) -> (B, N_ensemble, d_obs)
    sigma_y                 # scalar or (B,)
):
    """ Ensemble Randomized Square Root Filter (ETKF) - Analysis Step (Batched) """
    batch_size, N_ensemble, d_state = ensemble_f.shape
    device = ensemble_f.device
    dtype = ensemble_f.dtype

    if observation_y.ndim == 1:
        obs_y_eff = observation_y.unsqueeze(0) # (1, d_obs)
    else:
        obs_y_eff = observation_y # (B, d_obs)
        if obs_y_eff.shape[0] != batch_size and obs_y_eff.shape[0] != 1:
             raise ValueError("Batch size of observation_y must match ensemble_f or be 1.")

    # N1: (N_ensemble - 1)
    N1_val = max(N_ensemble - 1.0, 1.0) # scalar

    # ensemble_y_f: (B, N_ensemble, d_obs)
    ensemble_y_f = observation_operator_ens(ensemble_f)

    # Af: (B, N_ensemble, d_state), mean_f: (B, 1, d_state)
    Af, mean_f = center_ensemble(ensemble_f, rescale=False)
    # AYf: (B, N_ensemble, d_obs), mean_yf: (B, 1, d_obs)
    AYf, mean_yf = center_ensemble(ensemble_y_f, rescale=False)

    # Prepare sigma_y for division, ensure it's (B,1,1) or scalar
    if isinstance(sigma_y, torch.Tensor) and sigma_y.ndim > 0:
        sigma_y_sq_inv = (1.0 / sigma_y**2).view(-1, 1, 1) # (B,1,1) or (1,1,1)
    else: # scalar
        sigma_y_sq_inv = 1.0 / (sigma_y**2)

    # C_tilde_sym: (B, N_ensemble, N_ensemble)
    # Original AYf @ AYf.T is (N_ens, N_y) @ (N_y, N_ens) -> (N_ens, N_ens)
    # Batched: (B, N_ens, N_y) @ (B, N_y, N_ens) -> (B, N_ens, N_ens)
    C_tilde_sym = (AYf @ AYf.transpose(-2, -1)) * sigma_y_sq_inv + \
                  N1_val * torch.eye(N_ensemble, device=device, dtype=dtype).unsqueeze(0)

    eig_vals, eig_vecs = torch.linalg.eigh(C_tilde_sym) # eig_vals (B,N), eig_vecs (B,N,N)
    eig_vals_clamped = torch.clamp(eig_vals, min=1e-9)

    # T_transform_matrix: (B, N_ensemble, N_ensemble)
    T_transform_matrix = eig_vecs @ torch.diag_embed(eig_vals_clamped**-0.5) @ \
                         eig_vecs.transpose(-2, -1) * torch.sqrt(torch.tensor(N1_val, device=device, dtype=dtype))
    # Pw_term: (B, N_ensemble, N_ensemble)
    Pw_term = eig_vecs @ torch.diag_embed(eig_vals_clamped**-1) @ eig_vecs.transpose(-2, -1)

    # innovation_dy: (B, 1, d_obs)
    innovation_dy = (obs_y_eff.unsqueeze(1) - mean_yf)
    # w_gain_transpose: (B, 1, N_ensemble)
    w_gain_transpose = (innovation_dy @ AYf.transpose(-2, -1)) @ Pw_term * sigma_y_sq_inv

    # mean_a: (B, 1, d_state)
    mean_a = mean_f + w_gain_transpose @ Af
    # Af_updated: (B, N_ensemble, d_state)
    Af_updated = T_transform_matrix @ Af # T operates on rows of Af
    # ensemble_a: (B, N_ensemble, d_state)
    ensemble_a = mean_a + Af_updated

    return ensemble_a, None


def _letkf_core_etkf_update(
    local_E_f_mean,     # (B, N_x_local)
    local_A_f,          # (B, N_ens, N_x_local)
    eff_AY_f_anom,      # (B, N_ens, N_y_local)
    eff_d_f_innov,      # (B, N_y_local)
    N_ensemble
):
    """ Core ETKF update for local patches (Batched), assuming R_eff = I. """
    device = local_A_f.device; dtype = local_A_f.dtype
    batch_size = local_A_f.shape[0]
    N1_val = max(N_ensemble - 1.0, 1.0) # scalar

    # Pa_tilde_inv_sqrt: (B, N_ensemble, N_ensemble)
    Pa_tilde_inv_sqrt = eff_AY_f_anom @ eff_AY_f_anom.transpose(-2, -1) + \
                        N1_val * torch.eye(N_ensemble, device=device, dtype=dtype).unsqueeze(0)

    eig_vals, eig_vecs = torch.linalg.eigh(Pa_tilde_inv_sqrt)
    eig_vals_clamped = torch.clamp(eig_vals, min=1e-9)

    # T_transform: (B, N_ensemble, N_ensemble)
    T_transform = eig_vecs @ torch.diag_embed(eig_vals_clamped**-0.5) @ \
                  eig_vecs.transpose(-2, -1) * torch.sqrt(torch.tensor(N1_val, device=device, dtype=dtype))
    # Pw: (B, N_ensemble, N_ensemble)
    Pw = eig_vecs @ torch.diag_embed(eig_vals_clamped**-1) @ eig_vecs.transpose(-2, -1)

    # w_gain_transpose: (B, 1, N_ensemble)
    w_gain_transpose = (eff_d_f_innov.unsqueeze(1) @ eff_AY_f_anom.transpose(-2, -1)) @ Pw

    # local_mean_a: (B, N_x_local)
    # (w_gain_transpose @ local_A_f) is (B, 1, N_x_local)
    mean_update_term = (w_gain_transpose @ local_A_f).squeeze(1) # (B, N_x_local)
    local_mean_a = local_E_f_mean + mean_update_term

    # local_A_a: (B, N_ensemble, N_x_local)
    local_A_a = T_transform @ local_A_f

    return local_mean_a, local_A_a


def _letkf_analysis(
    ensemble_f,             # (B, N_ensemble, d_state)
    observation_y,          # (B, d_obs) or (d_obs,)
    observation_operator_ens, # (B, N_ensemble, d_state) -> (B, N_ensemble, d_obs)
    sigma_y,                # scalar observation error standard deviation
    localization_radius,    # scalar
    coords_state,           # (d_state, D_coord_state)
    coords_obs,             # (d_obs, D_coord_obs)
    domain=None     # (D_coord_state,) or (D_coord_obs,)
):
    """ Local Ensemble Transform Kalman Filter (LETKF) - Analysis Step (Batched) """
    batch_size, N_ensemble, d_state = ensemble_f.shape
    device = ensemble_f.device; dtype = ensemble_f.dtype

    if observation_y.ndim == 1:
        obs_y_eff = observation_y.unsqueeze(0) # (1, d_obs)
        d_obs = observation_y.shape[0]
    else:
        obs_y_eff = observation_y # (B, d_obs)
        d_obs = observation_y.shape[-1]
        if obs_y_eff.shape[0] != batch_size and obs_y_eff.shape[0] != 1:
             raise ValueError("Batch size of observation_y must match ensemble_f or be 1.")

    # ensemble_y_f: (B, N_ensemble, d_obs)
    ensemble_y_f = observation_operator_ens(ensemble_f)
    # Af_global: (B, N_ensemble, d_state), mean_f_global: (B, 1, d_state)
    Af_global, mean_f_global = center_ensemble(ensemble_f, rescale=False)
    # AYf_global: (B, N_ensemble, d_obs), mean_yf_global: (B, 1, d_obs)
    AYf_global, mean_yf_global = center_ensemble(ensemble_y_f, rescale=False)

    # innovation_mean_global: (B, 1, d_obs)
    innovation_mean_global = obs_y_eff.unsqueeze(1) - mean_yf_global

    # Transform observations and innovations by R^-1/2 (here R = sigma_y^2 * I)
    AYf_global_transformed = AYf_global / sigma_y         # (B, N_ensemble, d_obs)
    innovation_mean_global_transformed = innovation_mean_global / sigma_y # (B, 1, d_obs)

    # Initialize analysis ensemble parts
    ensemble_a_mean_parts = torch.zeros_like(mean_f_global) # (B, 1, d_state)
    ensemble_a_anom_parts = torch.zeros_like(Af_global)   # (B, N_ensemble, d_state)

    # Loop over each state variable to update it locally
    for k_state_idx in range(d_state):
        # current_mean_f_k: (B, 1) mean of k-th state var for all batches
        current_mean_f_k = mean_f_global[:, :, k_state_idx]
        # current_Af_k: (B, N_ensemble, 1) anomalies of k-th state var
        current_Af_k = Af_global[:, :, k_state_idx].unsqueeze(-1)

        # --- Localization: This part is NOT batched over `batch_size` ---
        # --- It's computed once per k_state_idx as coords are shared ---
        # Coords for k-th state var: (1, D_coord)
        coord_k_state = coords_state[k_state_idx].unsqueeze(0)

        # Distances from k-th state variable to all observations: (1, d_obs)
        # Assumes pairwise_distances can handle (N,D) (M,D) -> (N,M) inputs
        # or a specific 2D version is used for these non-batched coordinates.
        dist_state_k_to_obs = pairwise_distances(
            coord_k_state, coords_obs, domain=domain
        ).squeeze(0) # -> (d_obs,)

        rho_k = dist2coeff(dist_state_k_to_obs, localization_radius) # (d_obs,)
        local_obs_indices = torch.where(rho_k > 1e-6)[0] # (N_y_local_k,)

        if len(local_obs_indices) == 0: # No observations influence this state variable
            ensemble_a_mean_parts[:, :, k_state_idx] = current_mean_f_k
            ensemble_a_anom_parts[:, :, k_state_idx] = current_Af_k.squeeze(-1)
            continue

        # Select local observations for this k_state_idx
        # These are now batched over `batch_size`
        # AYf_local_k_transformed: (B, N_ensemble, N_y_local_k)
        AYf_local_k = AYf_global_transformed[:, :, local_obs_indices]
        # innov_local_k_transformed: (B, 1, N_y_local_k)
        innov_local_k = innovation_mean_global_transformed[:, :, local_obs_indices]

        # Apply localization weights to observations (sqrt_rho acts on transformed obs anoms)
        rho_local_k_weights = rho_k[local_obs_indices] # (N_y_local_k,)
        # sqrt_rho_local_k broadcastable: (1, 1, N_y_local_k)
        sqrt_rho_local_k_bcast = torch.sqrt(rho_local_k_weights).view(1, 1, -1)

        # eff_AYf_k_anom: (B, N_ensemble, N_y_local_k)
        eff_AYf_k_anom = AYf_local_k * sqrt_rho_local_k_bcast
        # eff_innov_k: (B, 1, N_y_local_k) -> squeezed to (B, N_y_local_k)
        eff_innov_k = (innov_local_k * sqrt_rho_local_k_bcast).squeeze(1)

        # Core ETKF update for (k_state_idx, and all batches)
        # current_mean_f_k is (B,1), current_Af_k is (B, N_ens, 1)
        updated_mean_k, updated_A_k = _letkf_core_etkf_update(
            current_mean_f_k, current_Af_k,
            eff_AYf_k_anom, eff_innov_k, N_ensemble
        )
        # updated_mean_k: (B,1), updated_A_k: (B, N_ensemble, 1)

        ensemble_a_mean_parts[:, :, k_state_idx] = updated_mean_k
        ensemble_a_anom_parts[:, :, k_state_idx] = updated_A_k.squeeze(-1)

    ensemble_a = ensemble_a_mean_parts + ensemble_a_anom_parts
    return ensemble_a, None

## iEnKS
def _ienks_analysis(
    ensemble_f,
    observation_y,
    observation_operator_ens,
    sigma_y,
    # --- Model specific args ---
    model_propagator,
    model_rhs,
    model_dt,
    # --- iEnKS hyperparameters ---
    upd_a='Sqrt', # <-- MODIFIED: Added update method selector
    Lag=1,
    nIter=10,
    wtol=1e-5,
    steps_between_analyses=5
):
    """
    Function:
        Implements a batched Iterative Ensemble Kalman Smoother (iEnKS) analysis step.
    """
    B, N, D_state = ensemble_f.shape
    device = ensemble_f.device
    dtype = ensemble_f.dtype

    if observation_y.ndim == 1:
        y = observation_y.unsqueeze(0)
    else:
        y = observation_y

    N1 = N - 1
    X0, x0 = center_ensemble(ensemble_f)

    w = torch.zeros(B, N, 1, device=device, dtype=dtype)
    T = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    Tinv = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    D_pert = None # For PertObs method

    if isinstance(sigma_y, torch.Tensor) and sigma_y.ndim > 0:
        R_inv_sqrt = (1.0 / sigma_y).view(-1, 1, 1)
    else:
        R_inv_sqrt = 1.0 / sigma_y

    def propagate_ensemble_in_window(ens_in):
        ens_flat = ens_in.view(B * N, D_state)
        num_model_steps = Lag * steps_between_analyses
        propagated_ens = ens_flat
        for _ in range(num_model_steps):
            propagated_ens = model_propagator(lambda x: model_rhs(x), propagated_ens, model_dt)
        return propagated_ens.view(B, N, D_state)

    for iteration in range(nIter):
        E_iter = x0 + T @ X0 + (X0.transpose(-1, -2) @ w).transpose(-1, -2)
        E_fwd = propagate_ensemble_in_window(E_iter)
        Eo = observation_operator_ens(E_fwd)

        Y, xo_obs = center_ensemble(Eo)
        dy_eff = (y.unsqueeze(1) - xo_obs) * R_inv_sqrt
        Y_eff = Y * R_inv_sqrt
        za = float(N1)

        Y_iter = Tinv @ Y_eff

        # Unified Cow1 calculation for all methods
        C_tilde = (Y_iter @ Y_iter.transpose(-2, -1)) + za * torch.eye(N, device=device, dtype=dtype)
        eig_vals, U = torch.linalg.eigh(C_tilde)
        eig_vals_clamped = torch.clamp(eig_vals, min=1e-9)
        Cow1 = U @ torch.diag_embed(1.0 / eig_vals_clamped) @ U.transpose(-2, -1)

        # Gauss-Newton optimization for weights `w`
        grad_term = Y_iter @ dy_eff.transpose(-2, -1)
        grad = grad_term - za * w
        dw = Cow1 @ grad

        # MODIFIED: Update transform matrices T and Tinv based on the chosen method
        if "Sqrt" in upd_a:
            eig_vals_sqrt = torch.sqrt(eig_vals_clamped)
            T = U @ torch.diag_embed(1.0 / eig_vals_sqrt) @ U.transpose(-2,-1) * math.sqrt(N1)
            Tinv = U @ torch.diag_embed(eig_vals_sqrt) @ U.transpose(-2,-1) / math.sqrt(N1)
        elif "PertObs" in upd_a:
            if iteration == 0:
                _D_pert = torch.randn_like(Y_eff)
                D_pert = _D_pert - _D_pert.mean(dim=1, keepdim=True)
            gradT = -(Y_eff + D_pert) @ Y_iter.transpose(-2, -1) + N1 * (torch.eye(N, device=device) - T)
            T = T + gradT @ Cow1
            Tinv = torch.linalg.inv(T + 1)
        elif "Order1" in upd_a:
            gradT = -0.5 * Y_eff @ Y_iter.transpose(-2, -1) + N1 * (torch.eye(N, device=device) - T)
            T = T + gradT @ Cow1
            Tinv = torch.linalg.inv(T)
        else:
            raise NotImplementedError(f"Update type '{upd_a}' not implemented.")

        w_new = w + dw
        if ((w_new - w).norm(p=2, dim=1)**2 / N).mean() < wtol:
            w = w_new
            break
        w = w_new

    final_delta_mean = (X0.transpose(-2, -1) @ w).transpose(-1, -2)
    final_X_smoothed = T @ X0
    E_smoothed_at_start = x0 + final_delta_mean + final_X_smoothed

    return E_smoothed_at_start, None


# EnKF analysis
def ensemble_kalman_filter_analysis(
    ensemble_f,             # (B, N_ensemble, d_state)
    observation_y,          # (B, d_obs) or (d_obs,) or None
    observation_operator_ens, # (B, N_ensemble, d_state) -> (B, N_ensemble, d_obs)
    sigma_y,                # scalar or (B,)
    method="EnKF-PertObs",
    inflation_factor=1.0,   # scalar
    # For EnKF-PertObs
    localization_matrix_Lxy=None, # (d_state, d_obs)
    localization_matrix_Lyy=None, # (d_obs, d_obs)
    # For LETKF
    localization_radius_letkf=None, # scalar
    coords_state_letkf=None,        # (d_state, D_coord)
    coords_obs_letkf=None,          # (d_obs, D_coord)
    domain_letkf=None,      # (D_coord,)
    # For iEnKS
    ienks_lag=1,
    ienks_niter=10,
    ienks_wtol=1e-5,
    model_args=None # Dict for model propagator info needed by iEnKS
):
    """ Main dispatcher for ensemble Kalman filter analysis (Batched) """
    kalman_gain_or_transform = None
    ensemble_a_raw = None

    if observation_y is None: # No observation, forecast is analysis
        ensemble_a_raw = ensemble_f
    elif method == "EnKF-PertObs":
        ensemble_a_raw, kalman_gain_or_transform = _enkf_pert_obs_analysis(
            ensemble_f, observation_y, observation_operator_ens, sigma_y,
            localization_matrix_Lxy, localization_matrix_Lyy
        )
    elif method == "ERSF": # ETKF variant
        ensemble_a_raw, kalman_gain_or_transform = _ersf_analysis(
            ensemble_f, observation_y, observation_operator_ens, sigma_y
        )
    elif method == "LETKF":
        if localization_radius_letkf is None or \
           coords_state_letkf is None or \
           coords_obs_letkf is None:
            raise ValueError("LETKF requires localization_radius, coords_state, and coords_obs.")
        ensemble_a_raw, kalman_gain_or_transform = _letkf_analysis(
            ensemble_f, observation_y, observation_operator_ens, sigma_y,
            localization_radius_letkf, coords_state_letkf,
            coords_obs_letkf, domain_letkf
        )
    elif method.startswith("iEnKS-"):
        if model_args is None:
            raise ValueError("iEnKS methods require 'model_args' dictionary.")

        # Extract update type from method name, e.g., "iEnKS-Sqrt" -> "Sqrt"
        try:
            update_type = method.split('-', 1)[1]
        except IndexError:
            raise ValueError(f"Invalid iEnKS method format: {method}. Expected 'iEnKS-UpdateType'.")

        ensemble_a_raw, kalman_gain_or_transform = _ienks_analysis(
            ensemble_f, observation_y, observation_operator_ens, sigma_y,
            model_propagator=model_args['propagator'],
            model_rhs=model_args['rhs'],
            model_dt=model_args['dt'],
            steps_between_analyses=model_args['steps_between_analyses'],
            upd_a=update_type, # Pass the extracted update type
            Lag=ienks_lag,
            nIter=ienks_niter,
            wtol=ienks_wtol,
        )
    else:
        raise ValueError(f"Unknown EnKF method: {method}")

    # Apply inflation to the raw analysis ensemble
    ensemble_analysis = apply_inflation(ensemble_a_raw, inflation_factor)

    return ensemble_analysis, kalman_gain_or_transform

# ##############################################################################
# # Lorenz 96 and RK4 for Testing
# ##############################################################################
def lorenz96_rhs(x, F=8):
    """
    Calculates the RHS of the Lorenz 96 equations.
    x can be a 1D tensor (D_state,) or a 2D tensor (batch_size, D_state).
    F is a scalar forcing term.
    """
    D = x.shape[-1] # Works for both 1D and 2D x

    # Efficiently calculate indices for all D components
    # Indices are relative to the current component 'k'
    # x_k-2, x_k-1, x_k, x_k+1
    # For dxdt[k] = (x[k+1] - x[k-2]) * x[k-1] - x[k] + F

    # Create rolled versions of x for vectorized computation
    # x_m2 means x[(i-2+D)%D] for each i
    # x_m1 means x[(i-1+D)%D] for each i
    # x_p1 means x[(i+1)%D] for each i
    x_m2 = torch.roll(x, shifts=2, dims=-1)
    x_m1 = torch.roll(x, shifts=1, dims=-1)
    x_p1 = torch.roll(x, shifts=-1, dims=-1)

    dxdt = (x_p1 - x_m2) * x_m1 - x + F
    return dxdt

def lorenz63_rhs(x, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    # Function: Calculates the RHS of the Lorenz 63 equations.
    # ---
    # Input:
    #   x (torch.Tensor): State tensor of shape (batch_size, 3) or (3,).
    #   sigma, rho, beta (float): Lorenz 63 parameters.
    # ---
    # Output:
    #   torch.Tensor: The derivatives (dx/dt, dy/dt, dz/dt) with the same shape as x.
    """
    # Unpack state variables
    x_val = x[..., 0]
    y_val = x[..., 1]
    z_val = x[..., 2]

    # Lorenz 63 equations
    dxdt = sigma * (y_val - x_val)
    dydt = x_val * (rho - z_val) - y_val
    dzdt = x_val * y_val - beta * z_val

    # Stack the results back into a tensor of the same shape as the input
    return torch.stack([dxdt, dydt, dzdt], dim=-1)

def rk4_step(rhs_func, x, dt):
    """
    Performs one RK4 step.
    x can be (D_state,) or (batch_size, D_state).
    rhs_func is compatible with batched x.
    """
    k1 = rhs_func(x)
    k2 = rhs_func(x + 0.5 * dt * k1)
    k3 = rhs_func(x + 0.5 * dt * k2)
    k4 = rhs_func(x + dt * k3)
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next

class Lorenz63:
    """
    Function:
        Implements the Lorenz 63 model with an RK4 stepper.
    """
    def __init__(self, sigma=10.0, rho=28.0, beta=8./3.):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def _rhs(self, x):
        """
        Function:
            Computes the right-hand side of the Lorenz 63 equations.
        Input:
            x (torch.Tensor): State vector(s) of shape [N, 3] or [3]. It can handle batches, e.g., [B*N, 3].
        Output:
            torch.Tensor: The derivative dx/dt for each state vector.
        """
        is_1d = x.ndim == 1
        if is_1d:
            x = x.unsqueeze(0)

        dxdt = torch.zeros_like(x)
        dxdt[:, 0] = self.sigma * (x[:, 1] - x[:, 0])
        dxdt[:, 1] = x[:, 0] * (self.rho - x[:, 2]) - x[:, 1]
        dxdt[:, 2] = x[:, 0] * x[:, 1] - self.beta * x[:, 2]

        return dxdt.squeeze(0) if is_1d else dxdt

    def step(self, rhs_func, x, dt):
        """
        Function:
            Advances the model state by one time step using RK4.
        Input:
            rhs_func (callable): The right-hand side function.
            x (torch.Tensor): Current state vector(s).
            dt (float): Time step size.
        Output:
            torch.Tensor: State vector(s) at the next time step.
        """
        k1 = rhs_func(x)
        k2 = rhs_func(x + dt * k1 / 2)
        k3 = rhs_func(x + dt * k2 / 2)
        k4 = rhs_func(x + dt * k3)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# ##############################################################################
# # Main Test Script
# ##############################################################################
if __name__ == '__main__':
    import argparse
    # --- 0. Command-Line Argument Parser
    # Setup argument parser to select DA methods to run.
    parser = argparse.ArgumentParser(description="Run Data Assimilation Benchmark")
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['iEnKS-PertObs', 'iEnKS-Sqrt', 'iEnKS-Order1'], # Default methods to run
        choices=['BPF', 'EnKF-PO', 'ERSF', 'iEnKS-PertObs', 'iEnKS-Sqrt', 'iEnKS-Order1'],
        help='A list of DA methods to run.'
    )
    args = parser.parse_args()
    methods_to_run = args.methods
    print(f"Running selected methods: {methods_to_run}")

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # -- 1. Experiment Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    dtype = torch.float32
    print(f"Using device: {device}, dtype: {dtype}")

    # --- System and DA Parameters ---
    lorenz_model = Lorenz63()
    dt = 0.03
    obs_every = 5
    total_obs = 400
    n_steps = total_obs * obs_every
    batch_size = 32

    params = {
        'N': 10,
        'Lag': 1,
        'nIter': 10,
        'wtol': 1e-5,
        'infl': 1.1,
        'obs_noise_std': 1.0,
        'model_noise_std': 0.0
    }

    # --- Observation Setup ---
    obs_inds = [0] # Observe only the 'x' variable
    obs_operator = lambda x: x[..., obs_inds]
    l63_rhs_func = lambda x: lorenz_model._rhs(x)
    rk4_stepper = lambda rhs, x, dt: lorenz_model.step(rhs, x, dt)

    print("Generating true state with model spin-up...")
    x_spinup = torch.randn(batch_size, 3, device=device, dtype=torch.float32)
    num_spinup_steps = 2000
    for _ in range(num_spinup_steps):
        x_spinup = rk4_stepper(l63_rhs_func, x_spinup, dt)
    print("Spin-up complete.")

    print("Generating full true trajectory batch...")
    x_true = torch.zeros((batch_size, n_steps + 1, 3), device=device, dtype=torch.float32)
    x_true[:, 0] = x_spinup
    for i in range(n_steps):
        x_true[:, i+1] = rk4_stepper(l63_rhs_func, x_true[:, i], dt)

    obs_time_indices = torch.arange(0, n_steps + 1, obs_every)
    xx_true_obs = x_true[:, obs_time_indices]

    true_obs_vals = xx_true_obs[:, :, obs_inds]
    yy = true_obs_vals + torch.randn_like(true_obs_vals) * params['obs_noise_std']

    # -- 3. Initialize Ensembles for All Methods
    print("Initializing ensembles for selected methods...")
    # All methods start from the same initial state
    initial_true_state_b = xx_true_obs[:, 0, :].unsqueeze(1)
    noise = torch.randn(batch_size, params['N'], 3, device=device) * 1.0
    initial_ensemble = initial_true_state_b + noise

    # Initialize dictionaries for results and ensembles based on selected methods
    results_rmse = {method: [] for method in methods_to_run}
    analysis_times = {method: [] for method in methods_to_run}
    ensemble_states = {method: initial_ensemble.clone() for method in methods_to_run}

    # -- 4. Run Assimilation Loop
    print(f"Running {total_obs} analysis cycles...")
    for ko in tqdm(range(total_obs), desc="DA Benchmark"):

        # --- A: Standard Filter Implementations ---
        # 1. Forecast step for filters
        filter_forecasts = {}
        for name in methods_to_run:
            # Skip iEnKS here, as its forecast logic is handled differently
            if name.startswith('iEnKS'):
                continue

            ens = ensemble_states[name]
            ens_flat = ens.view(-1, 3) # Shape: [batch_size * N, 3]
            for _ in range(obs_every):
                ens_flat = rk4_stepper(l63_rhs_func, ens_flat, dt)
            forecast = ens_flat.view(batch_size, params['N'], 3)
            forecast += torch.randn_like(forecast) * params['model_noise_std']
            filter_forecasts[name] = forecast

        # 2. Analysis step for filters
        y_current = yy[:, ko, :]

        if 'BPF' in methods_to_run:
            start_time = time.perf_counter()
            analysis_bpf = bootstrap_particle_filter_analysis(
                filter_forecasts["BPF"], y_current, obs_operator, params['obs_noise_std'], resampling_method="systematic"
            )
            analysis_times["BPF"].append(time.perf_counter() - start_time)
            results_rmse["BPF"].append(torch.sqrt(torch.mean((analysis_bpf.mean(dim=1) - xx_true_obs[:, ko, :])**2, dim=-1)).cpu())
            ensemble_states["BPF"] = analysis_bpf

        common_enkf_args = {
            "observation_y": y_current,
            "observation_operator_ens": obs_operator,
            "sigma_y": params['obs_noise_std'],
            "inflation_factor": params['infl']
        }

        if 'EnKF-PO' in methods_to_run:
            start_time = time.perf_counter()
            analysis_enkf_po, _ = ensemble_kalman_filter_analysis(filter_forecasts["EnKF-PO"], method="EnKF-PertObs", **common_enkf_args)
            analysis_times["EnKF-PO"].append(time.perf_counter() - start_time)
            results_rmse["EnKF-PO"].append(torch.sqrt(torch.mean((analysis_enkf_po.mean(dim=1) - xx_true_obs[:, ko, :])**2, dim=-1)).cpu())
            ensemble_states["EnKF-PO"] = analysis_enkf_po

        if 'ERSF' in methods_to_run:
            start_time = time.perf_counter()
            analysis_ersf, _ = ensemble_kalman_filter_analysis(filter_forecasts["ERSF"], method="ERSF", **common_enkf_args)
            analysis_times["ERSF"].append(time.perf_counter() - start_time)
            results_rmse["ERSF"].append(torch.sqrt(torch.mean((analysis_ersf.mean(dim=1) - xx_true_obs[:, ko, :])**2, dim=-1)).cpu())
            ensemble_states["ERSF"] = analysis_ersf

        # --- B: iEnKS Implementation (Smoother Logic) ---
        ienks_methods = [m for m in methods_to_run if m.startswith('iEnKS-')]

        if ienks_methods:
            # This setup is common for all iEnKS methods
            k_start = max(0, ko - params['Lag'])

            model_args_ienks = {
                "propagator": rk4_stepper,
                "rhs": l63_rhs_func,
                "dt": dt,
                "steps_between_analyses": obs_every,
            }

            # Loop through each specific iEnKS method (e.g., iEnKS-Sqrt, iEnKS-PertObs)
            for method_name in ienks_methods:
                # 1. Analysis step for the specific iEnKS method
                start_time = time.perf_counter()
                E_smoothed_at_start, _ = ensemble_kalman_filter_analysis(
                    ensemble_f=ensemble_states[method_name], # Use the forecast for this method
                    observation_y=y_current,
                    observation_operator_ens=obs_operator,
                    sigma_y=params['obs_noise_std'],
                    method=method_name, # Pass the specific method name
                    inflation_factor=params['infl'],
                    ienks_lag=(ko - k_start),
                    ienks_niter=params['nIter'],
                    ienks_wtol=params['wtol'],
                    model_args=model_args_ienks
                )
                analysis_times[method_name].append(time.perf_counter() - start_time)

                # 2. Propagate smoothed state to current time `ko` for RMSE calculation
                E_analysis_at_ko = E_smoothed_at_start.clone()
                num_steps_to_propagate = (ko - k_start) * obs_every
                if num_steps_to_propagate > 0:
                    E_flat = E_analysis_at_ko.view(-1, 3)
                    for _ in range(num_steps_to_propagate):
                        E_flat = rk4_stepper(l63_rhs_func, E_flat, dt)
                    E_analysis_at_ko = E_flat.view(batch_size, params['N'], 3)

                # Store results for the current method
                rmse = torch.sqrt(torch.mean((E_analysis_at_ko.mean(dim=1) - xx_true_obs[:, ko, :])**2, dim=-1))
                results_rmse[method_name].append(rmse.cpu())

                # 3. Create the forecast for the *next* cycle's window start
                E_flat_next = E_smoothed_at_start.view(-1, 3)
                for _ in range(obs_every):
                    E_flat_next = rk4_stepper(l63_rhs_func, E_flat_next, dt)
                forecast_next = E_flat_next.view(batch_size, params['N'], 3)

                # Add model noise for the next forecast
                forecast_next += torch.randn_like(forecast_next) * params['model_noise_std']

                # Update the state for the current method
                ensemble_states[method_name] = forecast_next


    # -- 5. Final Results
    print("\n--- Final Average Metrics (Stable Period) ---")
    for method_name in methods_to_run:
        rmse_tensor = torch.stack(results_rmse[method_name])
        time_tensor = torch.tensor(analysis_times[method_name])
        print("RMSE tensor shape:", rmse_tensor.shape)
        stable_start_idx = 0
        rmse_mean = rmse_tensor[stable_start_idx:, :].mean().item()
        avg_time = time_tensor[stable_start_idx:].mean().item()

        print(f"{method_name:<8s}: Avg. Time: {avg_time:.6f} s | Mean RMSE (stable): {rmse_mean:.4f}")
