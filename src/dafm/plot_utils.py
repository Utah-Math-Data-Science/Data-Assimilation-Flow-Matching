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
    Plot particle spreads & means for each method in cfgs.

    Parameters
    ----------
    cfgs : dict
        Mapping keys → config dict v.  Each v must contain:
          - v['trajectories']: DataFrame indexed by time, with columns
            * predicted_state_{i}_dim_{d}
            * true_state_dim_{d} (or any non-predicted column)
    dims : list[int], optional
        Which dims to plot; if None, infer from the first v.
    mode : {'width','color','no'}
        'width' or 'color' draw spreads; 'no' only draws mean lines.
    hist_bins : int
        Number of bins per time slice.
    hist_step : int
        Plot every Nth slice.
    max_time_steps : int, optional
        Only plot up to this many steps (truncated to data length).
    save_fig : bool
        If True, save PNGs instead of showing.
    save_name : str
        Base filename for saving (method & dim appended).
    """

    # 1) collect & print all method names
    method_names = []
    for key in cfgs:
        if isinstance(key, tuple) and len(key) > 1:
            name = key[1]
        else:
            name = str(key)
        method_names.append(name)
    print("Methods found:", method_names)

    # 2) loop per method
    for key, v in cfgs.items():
        # determine method name from key
        if isinstance(key, tuple) and len(key) > 1:
            method_name = key[1]
        else:
            method_name = str(key)

        # reshape trajectories
        df = v['trajectories'].copy()
        df.index.name = 'Time'
        df = df.reset_index()
        cols = df.columns

        # detect predicted vs true columns
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

        # infer dims if not provided
        found_dims = sorted(
            int(re.search(r'_dim_(\d+)$', s).group(1))
            for s in pred_cols if re.search(r'_dim_(\d+)$', s)
        )
        plot_dims = dims if dims is not None else found_dims

        # per-dimension plots
        for dim in plot_dims:
            df_h = df_hist[df_hist['Source'].str.endswith(f'_dim_{dim}')]
            df_t = df_true[df_true['Source'].str.endswith(f'_dim_{dim}')]

            times = np.sort(df_h['Time'].unique())
            if max_time_steps is not None:
                times = times[:max_time_steps]
            df_h = df_h[df_h['Time'].isin(times)]
            df_t = df_t[df_t['Time'].isin(times)]

            # precompute for color mode
            if mode == 'color':
                masses = []
                for t in times[::hist_step]:
                    vals = df_h.loc[df_h['Time'] == t, 'State'].values
                    h, b = np.histogram(vals, bins=hist_bins, density=True)
                    masses.extend(h * np.diff(b))
                global_max = max(masses) if masses else 1.0

            # start figure
            plt.figure(figsize=(12, 6))

            # plot true‐state mean
            true_mean = df_t.groupby('Time')['True'].mean()
            plt.plot(
                true_mean.index, true_mean.values,
                label='True state', color='tab:gray', linewidth=1
            )

            # plot particle mean
            pred_mean = df_h.groupby('Time')['State'].mean()
            plt.plot(
                pred_mean.index, pred_mean.values,
                label='Particle mean', color='red', linestyle='--', linewidth=1
            )

            # overlay spread if requested
            if mode != 'no':
                for t in times[::hist_step]:
                    vals = df_h.loc[df_h['Time'] == t, 'State'].values
                    h, b = np.histogram(vals, bins=hist_bins, density=True)
                    centers = 0.5 * (b[:-1] + b[1:])

                    if mode == 'width':
                        scale = h.max() or 1.0
                        widths = h / scale * 0.8
                        plt.fill_betweenx(
                            centers,
                            t - widths, t + widths,
                            facecolor='orange', edgecolor='none', alpha=0.5
                        )
                    else:  # color
                        cmap = cm.get_cmap('Oranges')
                        for mass, start, end in zip(h * np.diff(b), b[:-1], b[1:]):
                            intensity = 0.2 + 0.5 * (mass / global_max)
                            color = cmap(intensity)
                            plt.fill_between(
                                [t - 0.5, t + 0.5],
                                [start, start],
                                [end, end],
                                color=color, linewidth=0
                            )

            plt.xlabel('Time')
            plt.ylabel(f'dim {dim}')
            plt.title(method_name)
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.tight_layout()

            if save_fig:
                fname = f"{save_name}_{method_name}_dim_{dim}_{mode}.png"
                plt.savefig(fname, dpi=150)
                plt.close()
            else:
                plt.show()
