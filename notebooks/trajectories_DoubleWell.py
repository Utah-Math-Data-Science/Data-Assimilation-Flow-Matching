import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    from collections import defaultdict
    import copy
    import os
    import pprint

    import hydra
    from omegaconf import OmegaConf
    from einops import rearrange
    import numpy as np
    import pandas as pd
    import seaborn as sns
    sns.set_theme(style='whitegrid', font_scale=1.3, palette=sns.color_palette('Set2'),)
    import sqlalchemy as sa
    import marimo as mo
    import seaborn as sns

    from conf import conf
    from dafm import datasets, models, plots, utils
    return (
        OmegaConf,
        conf,
        copy,
        datasets,
        defaultdict,
        hydra,
        mo,
        models,
        np,
        os,
        pd,
        plots,
        pprint,
        rearrange,
        sa,
        sns,
        utils,
    )


@app.cell
def _():
    alt_ids = {
        ('z6by8yki', r'SM-TF ($\epsilon_\alpha=0.5$, $\epsilon_\beta=0.025$)'): {},
        ('0jhhvwaj', r'FM-TF ($\alpha=0.0$, $\sigma_\epsilon=0.0$)'): {},
    }
    alt_id_to_label = dict(list(alt_ids))
    label_to_alt_id = dict(map(reversed, alt_ids))
    assert len(alt_ids) == len(label_to_alt_id), "Do two alt_id's have the same plot label?"
    return alt_id_to_label, alt_ids, label_to_alt_id


@app.cell
def _(alt_ids, conf, pprint, sa):
    engine = conf.get_engine()
    with conf.sa.orm.Session(engine) as db:
        cfgs = db.execute(sa.select(conf.Conf).where(conf.Conf.alt_id.in_([k[0] for k in alt_ids])))
        cfgs = {c.alt_id: c for (c,) in cfgs}
        cfgs = {k: {'cfg': cfgs[k[0]]} for k in alt_ids}
        pprint.pp(cfgs)
    return cfgs, db, engine


@app.cell
def _(cfgs, pd):
    predicted_state_columns = None
    true_state_columns = None
    for k, v in cfgs.items():
        v['trajectories'] = (
            pd.read_parquet(v['cfg'].run_dir/v['cfg'].prediction_filename)
        )
        v['trajectories']['alt_id'] = k[0]
        v['trajectories']['Model'] = k[1]
        if predicted_state_columns is None:
            predicted_state_columns = v['trajectories'].columns[
                v['trajectories'].columns.str.startswith('predicted_state_')
            ]
        if true_state_columns is None:
            true_state_columns = v['trajectories'].columns[
                v['trajectories'].columns.str.startswith('true_state_')
            ]
    cfgs[k]
    return k, predicted_state_columns, true_state_columns, v


@app.cell
def _(cfgs, mo, pd, predicted_state_columns):
    predicted_state_trajectory = (
        pd.concat([
            v['trajectories'][['alt_id', 'Model', 'times', *predicted_state_columns]]
            for v in cfgs.values()
        ])
        .melt(id_vars=['alt_id', 'Model', 'times'], var_name='ParticleAndDimension', value_name='State')
    )
    particle_and_dimension = predicted_state_trajectory['ParticleAndDimension'].str.split('_')
    predicted_state_trajectory['Particle'] = particle_and_dimension.str[2].map(int)
    predicted_state_trajectory['Dimension'] = particle_and_dimension.str[4].map(int)
    predicted_state_trajectory = (
        predicted_state_trajectory.set_index(['alt_id', 'Model', 'times', 'Dimension'])
        .pivot(columns='Particle', values='State')
    )
    predicted_state_trajectory['Mean'] = predicted_state_trajectory.mean(1)
    mo.plain(predicted_state_trajectory)
    return particle_and_dimension, predicted_state_trajectory


@app.cell
def _(mo, true_state_columns, v):
    true_state_trajectory = (
        v['trajectories'][['times', *true_state_columns]]
        .melt(id_vars='times', var_name='Dimension', value_name='State')
    )
    true_state_trajectory['Particle'] = 'True'
    true_state_trajectory['Dimension'] = true_state_trajectory['Dimension'].str.split('_').str[3].map(int)
    true_state_trajectory = (
        true_state_trajectory.set_index(['times', 'Dimension'])
        .pivot(columns='Particle', values='State')
    )
    mo.plain(true_state_trajectory)
    return (true_state_trajectory,)


@app.cell
def _(cfgs, predicted_state_trajectory, true_state_trajectory):
    (
        predicted_state_trajectory['Mean']
        .groupby(level=['alt_id', 'Model'])
        # compute L2 / sqrt(dim)
        .apply(lambda x: x.loc[x.name] - true_state_trajectory['True'])
        .pow(2)
        .groupby(level=['alt_id', 'Model', 'times'])
        .mean()
        .pow(1/2)
        # mean over trajectory
        .groupby(level=['alt_id', 'Model'])
        .mean()
        # sort
        .loc[[k[0] for k in cfgs]]
        .rename('Mean(time) RMSE(dim)')
        .to_frame()
    )
    return


@app.cell
def _(alt_id_to_label, label_to_alt_id):
    dim_to_plot = 0
    plot_identifier = 'Model'
    if plot_identifier == 'alt_id':
        row_order = list(alt_id_to_label)
    elif plot_identifier == 'Model':
        row_order = list(label_to_alt_id)
    else:
        raise ValueError(f'Unknown plot identifier: {plot_identifier}')
    hue_order = row_order
    return dim_to_plot, hue_order, plot_identifier, row_order


@app.cell
def _(
    dim_to_plot,
    hue_order,
    plot_identifier,
    predicted_state_trajectory,
    row_order,
    sns,
    true_state_trajectory,
):
    plot_predicted = (
        sns.relplot(
            kind='line',
            data=predicted_state_trajectory.loc[(slice(None), slice(None), slice(None), dim_to_plot), 'Mean'].reset_index(),
            x='times',
            y='Mean',
            row=plot_identifier,
            row_order=row_order,
            style=plot_identifier,
            hue=plot_identifier,
            hue_order=hue_order,
            markers=True,
            aspect=3,
        )
    )
    plot_predicted.map(
        sns.lineplot,
        data=true_state_trajectory.loc[(slice(None), 0), 'True'].reset_index(),
        x='times',
        y='True',
        color='tab:gray',
        zorder=0,
    )
    sns.move_legend(
        plot_predicted,
        loc='upper center',
        ncol=min(len(predicted_state_trajectory.index.get_level_values(plot_identifier).unique()), 2) + 1,
        title='',
        bbox_to_anchor=(.455, 1.06),
        frameon=True,
        fancybox=True,
    )
    plot_predicted
    return (plot_predicted,)


@app.cell
def _(
    dim_to_plot,
    hue_order,
    plot_identifier,
    predicted_state_trajectory,
    row_order,
    sns,
    true_state_trajectory,
):
    plot_histogram = (
        sns.displot(
            data=(
                predicted_state_trajectory
                .loc[(slice(None), slice(None), slice(None), dim_to_plot), :]
                .filter(regex='\d+', axis=1)
                .melt(var_name='Particle', value_name='State', ignore_index=False)
                .reset_index()
            ),
            x='times',
            y='State',
            bins=(true_state_trajectory.index.get_level_values('times').unique(), 101),
            row=plot_identifier,
            row_order=row_order,
            hue=plot_identifier,
            hue_order=hue_order,
            aspect=3,
            zorder=0,
        )
    )
    plot_histogram.map(
        sns.lineplot,
        data=true_state_trajectory.loc[(slice(None), dim_to_plot), 'True'].reset_index(),
        x='times',
        y='True',
        color='tab:gray',
        markers=True,
    )
    sns.move_legend(
        plot_histogram,
        loc='upper center',
        ncol=min(len(predicted_state_trajectory.index.get_level_values('alt_id').unique()), 2) + 1,
        title='',
        bbox_to_anchor=(.455, 1.06),
        frameon=True,
        fancybox=True,
    )
    plot_histogram
    return (plot_histogram,)


@app.cell
def _(
    dim_to_plot,
    hue_order,
    plot_identifier,
    predicted_state_trajectory,
    row_order,
    sns,
    true_state_trajectory,
):
    plot_confidence = (
        sns.relplot(
            kind='line',
            errorbar=('ci', 95),
            data=(
                predicted_state_trajectory
                .loc[(slice(None), slice(None), slice(None), dim_to_plot), :]
                .filter(regex='\d+', axis=1)
                .melt(var_name='Particle', value_name='State', ignore_index=False)
                .reset_index()
            ),
            x='times',
            y='State',
            row=plot_identifier,
            row_order=row_order,
            hue=plot_identifier,
            hue_order=hue_order,
            aspect=3,
            zorder=0,
        )
    )
    plot_confidence.map(
        sns.lineplot,
        data=true_state_trajectory.loc[(slice(None), dim_to_plot), 'True'].reset_index(),
        x='times',
        y='True',
        color='tab:gray',
        markers=True,
    )
    sns.move_legend(
        plot_confidence,
        loc='upper center',
        ncol=min(len(predicted_state_trajectory.index.get_level_values('alt_id').unique()), 2) + 1,
        title='',
        bbox_to_anchor=(.455, 1.06),
        frameon=True,
        fancybox=True,
    )
    plot_confidence
    return (plot_confidence,)


@app.cell(disabled=True)
def _(label_to_alt_id, plot_histogram, plot_predicted, plots):
    plots.save_all_subfigures(plot_predicted, 'Predicted', renaming=label_to_alt_id)
    plots.save_all_subfigures(plot_histogram, 'PredictedStateHistogram', renaming=label_to_alt_id)
    return


if __name__ == "__main__":
    app.run()
