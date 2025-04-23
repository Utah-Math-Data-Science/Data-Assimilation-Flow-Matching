import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_particle_trajectories_with_histograms(
    cfgs: dict,
    dims: list = None,
    mode: str = 'width',
    hist_bins: int = 15,
    hist_step: int = 1,
    max_time_steps: int = None,
    save_fig: bool = False,
    save_name: str = 'example_fig'
):
    """
    Plot particle spreads & means for each method in cfgs, using adaptive bar widths,
    and overlay observation trajectories for observed dimensions.

    Parameters
    ----------
    cfgs : dict
        Keys are method identifiers, values are dicts with:
          - 'cfg': configuration object with dataset.observe.n
          - 'trajectories': DataFrame indexed by time, columns including
            predicted_state_{i}_dim_{d}, true_state columns, and observation_dim_{k}
    dims : list[int], optional
        Which dims to plot; if None, infer from the first v.
    mode : {'width','color','no'}
        'width' or 'color' draw spreads; 'no' only draws mean lines.
    hist_bins : int
        Number of bins per dimension (global over all times).
    hist_step : int
        Plot every Nth time slice.
    max_time_steps : int, optional
        Cap to first N time steps.
    save_fig : bool
        If True, save PNGs instead of showing.
    save_name : str
        Base filename for saving (method & dim appended).
    """

    # 1) collect & print all method names
    methods = []
    for key in cfgs:
        if isinstance(key, tuple) and len(key) > 1:
            methods.append(key[1])
        else:
            methods.append(str(key))
    print("Methods found:", methods)

    # 2) loop per method
    for key, v in cfgs.items():
        # determine method name
        if isinstance(key, tuple) and len(key) > 1:
            method_name = key[1]
        else:
            method_name = str(key)

        # prepare config and observation interval
        cfg = v.get('cfg', None)
        if cfg is not None and hasattr(cfg, 'dataset') and hasattr(cfg.dataset, 'observe'):
            n_obs = getattr(cfg.dataset.observe, 'n', None)
        else:
            n_obs = None

        # prepare the trajectories DataFrame
        df = v['trajectories'].copy()
        df.index.name = 'Time'
        df = df.reset_index()
        cols = df.columns

        # classify predicted vs. true columns by regex
        pred_pat = re.compile(r'^predicted_state_\d+_dim_\d+$')
        true_pat = re.compile(r'^(true_state|state)(_dim_\d+)?$')
        pred_cols = [c for c in cols if pred_pat.match(c)]
        true_cols = [c for c in cols if true_pat.match(c)]
        if not true_cols:
            true_cols = [c for c in cols if c not in pred_cols and c != 'Time']

        # melt to long form
        df_hist = df.melt(
            id_vars=['Time'],
            value_vars=pred_cols,
            var_name='Source', value_name='State'
        )
        df_true = df.melt(
            id_vars=['Time'],
            value_vars=true_cols,
            var_name='Source', value_name='True'
        )

        # infer dimensions if not provided
        found_dims = sorted(
            int(re.search(r'_dim_(\d+)$', src).group(1))
            for src in pred_cols if re.search(r'_dim_(\d+)$', src)
        )
        plot_dims = dims if dims is not None else found_dims

        # 3) per-dimension plotting
        for dim in plot_dims:
            # subset to this dimension
            sub = df_hist[df_hist['Source'].str.endswith(f'_dim_{dim}')]
            tpl = df_true[df_true['Source'].str.endswith(f'_dim_{dim}')]

            # global bins for this dimension
            all_states = sub['State'].values
            if len(all_states) == 0:
                continue
            gmin, gmax = all_states.min(), all_states.max()
            if np.isclose(gmin, gmax):
                bins = np.array([gmin - 0.5, gmax + 0.5])
            else:
                bins = np.linspace(gmin, gmax, hist_bins + 1)

            # time axis and truncation
            times = np.sort(sub['Time'].unique())
            if max_time_steps is not None:
                times = times[:max_time_steps]

            # adaptive half-width from actual time spacing
            if len(times) > 1:
                dt = np.min(np.diff(times))
            else:
                dt = 1.0
            half_width = dt / 2.0

            # filter to selected times
            sub = sub[sub['Time'].isin(times)]
            tpl = tpl[tpl['Time'].isin(times)]

            # start figure
            plt.figure(figsize=(12, 6))

            # plot true-state mean
            true_mean = tpl.groupby('Time')['True'].mean()
            plt.plot(
                true_mean.index, true_mean.values,
                label='True state', color='tab:gray', linewidth=2
            )

            # plot particle mean
            pred_mean = sub.groupby('Time')['State'].mean()
            plt.plot(
                pred_mean.index, pred_mean.values,
                label='Particle mean', color='red',
                linestyle='--', linewidth=2
            )

            # plot observation if this dim is observed
            if n_obs is not None and dim % n_obs == 0:
                obs_idx = dim // n_obs
                obs_col = f'observation_dim_{obs_idx}'
                if obs_col in cols:
                    df_obs = df[['Time', obs_col]].copy()
                    df_obs = df_obs[df_obs['Time'].isin(times)]
                    plt.plot(
                        df_obs['Time'], df_obs[obs_col],
                        label='Observation', color='blue',
                        linewidth=0.5, marker='o', markersize=4
                    )

            # overlay spread unless mode='no'
            if mode != 'no':
                # precompute global max mass for color mode
                if mode == 'color':
                    masses = []
                    for t in times[::hist_step]:
                        vals = sub.loc[sub['Time'] == t, 'State'].values
                        h, _ = np.histogram(vals, bins=bins, density=True)
                        masses.extend(h * np.diff(bins))
                    global_max = max(masses) if masses else 1.0

                for t in times[::hist_step]:
                    vals = sub.loc[sub['Time'] == t, 'State'].values
                    h, _ = np.histogram(vals, bins=bins, density=True)
                    centers = 0.5 * (bins[:-1] + bins[1:])
                    hist_max = h.max() + 1e-12

                    if mode == 'width':
                        widths = (h / hist_max) * 0.8
                        plt.fill_betweenx(
                            centers,
                            t - widths * half_width,
                            t + widths * half_width,
                            facecolor='orange', edgecolor='none', alpha=0.5
                        )
                    else:  # color
                        cmap = cm.get_cmap('Oranges')
                        for mass, y0, y1 in zip(h * np.diff(bins), bins[:-1], bins[1:]):
                            if mass <= 0:
                                continue
                            intensity = 0.2 + 0.5 * (mass / global_max)
                            col = cmap(intensity)
                            plt.fill_between(
                                [t - half_width, t + half_width],
                                [y0, y0], [y1, y1],
                                color=col, alpha=0.5, linewidth=0
                            )

            plt.xlabel('Time')
            plt.ylabel(f'dim {dim}')
            plt.title(method_name)
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.tight_layout()

            if save_fig:
                fn = f"{save_name}_{method_name}_dim_{dim}_{mode}.png"
                plt.savefig(fn, dpi=150)
                plt.close()
            else:
                plt.show()
