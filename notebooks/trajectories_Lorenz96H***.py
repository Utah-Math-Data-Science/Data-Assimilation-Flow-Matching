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
    import polars as pl

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
        pl,
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
        ('bz6r81zr', r'SMM (2376999025)'): {},
        ('806lerm0', r'SMM (649520)'): {},
        ('p3s7zz42', r'SMM (5113685)'): {},
        ('1xb6wto5', r'SMM (5543464)'): {},
        ('6ga8dhz9', r'SMM (1663576)'): {},
        ('7kmx2xcx', r'SMM (1013721)'): {},
        ('7hfcetdp', r'SMM (2347148)'): {},
        ('xe4kqnc2', r'SMM (4141989)'): {},
        ('wt5wrr1k', r'SMM (179266)'): {},
        ('6o2xdada', r'SMM (4824560)'): {},
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
def _(cfgs, pl):
    trajectories = []
    for k, v in cfgs.items():
        trajectories.append(
            pl.scan_parquet(v['cfg'].run_dir/v['cfg'].prediction_filename)
            .select(
                pl.col('*'),
                alt_id=pl.lit(k[0]),
                Model=pl.lit(k[1]),
            )
        )
    trajectories = pl.concat(trajectories)
    return k, trajectories, v


@app.cell
def _(pl, trajectories):
    predicted_state_trajectory = (
        trajectories.unpivot(
            on=pl.selectors.starts_with('predicted_state_'),
            index=[
                pl.col('alt_id'), pl.col('Model'),
                pl.col('times'),
            ],
            variable_name='ParticleAndDimension',
            value_name='State',
        )
        .select(
            pl.col('alt_id'), pl.col('Model'),
            pl.col('times'),
            pl.col('State'),
            Particle=pl.col('ParticleAndDimension').str.split('_').list[2].cast(int),
            Dimension=pl.col('ParticleAndDimension').str.split('_').list[4].cast(int),
            # Mean=pl.col('State').mean(1)
        )
    )
    predicted_state_trajectory
    return (predicted_state_trajectory,)


@app.cell
def _(pl, trajectories):
    true_state_trajectory = (
        trajectories.unpivot(
            on=pl.selectors.starts_with('true_state_'),
            index=[
                pl.col('alt_id'), pl.col('Model'),
                pl.col('times'),
            ],
            variable_name='Dimension',
            value_name='State',
        )
        .select(
            pl.col('alt_id'), pl.col('Model'),
            pl.col('times'),
            pl.col('State'),
            Particle=pl.lit('True'),
            Dimension=pl.col('Dimension').str.split('_').list[3].cast(int),
        )
    )
    true_state_trajectory
    return (true_state_trajectory,)


@app.cell
def _(pl, trajectories):
    observation_trajectory = (
        trajectories.unpivot(
            on=pl.selectors.starts_with('observation_'),
            index=[
                pl.col('alt_id'), pl.col('Model'),
                pl.col('times'),
            ],
            variable_name='Dimension',
            value_name='State',
        )
        .select(
            pl.col('alt_id'), pl.col('Model'),
            pl.col('times'),
            pl.col('State'),
            Particle=pl.lit('Observation'),
            Dimension=pl.col('Dimension').str.split('_').list[2].cast(int),
        )
    )
    observation_trajectory
    return (observation_trajectory,)


@app.cell
def _(pl, predicted_state_trajectory, true_state_trajectory):
    rmse = (
        predicted_state_trajectory
        .group_by('alt_id', 'Model', 'times', 'Dimension')
        .agg(pl.col('State').mean().alias('Mean'))
        .join(
            true_state_trajectory,
            on=['alt_id', 'Model', 'times', 'Dimension'],
        )
        .select(
            pl.col('alt_id'), pl.col('Model'),
            pl.col('times'),
            pl.col('Dimension'),
            (pl.col('Mean') - pl.col('State')).pow(2).alias('DiffPow2'),
        )
        .group_by('alt_id', 'Model', 'times')
        .agg(pl.col('DiffPow2').mean().pow(1/2).alias('RMSE(dim)'))

        # take last time steps
        .sort('times')
        .group_by('alt_id', 'Model')
        .tail(50)
    
        .group_by('alt_id', 'Model')
        .agg(pl.col('RMSE(dim)').mean().alias('Mean(time) RMSE(dim)'))

        .collect()
    )
    print(rmse.mean())
    rmse
    return (rmse,)


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
def _(dim_to_plot, plot_identifier, sns, true_state_trajectory):
    def map_true_state_trajectory(plot):
        for (row, col, hue), _ in plot.facet_data():
            ax = plot.axes[row][col]
            data = (
                true_state_trajectory
                .filter(**{
                    'Dimension': dim_to_plot,
                    plot_identifier: plot.row_names[row],
                })
                .collect()
                .to_pandas()
            )
            sns.lineplot(
                data=data,
                x='times',
                y='State',
                color='tab:gray',
                ax=ax,
                legend=False,
            )
    return (map_true_state_trajectory,)


@app.cell
def _(
    alt_ids,
    dim_to_plot,
    hue_order,
    map_true_state_trajectory,
    pl,
    plot_identifier,
    predicted_state_trajectory,
    row_order,
    sns,
):
    plot_predicted = (
        sns.relplot(
            kind='line',
            data=(
                predicted_state_trajectory
                .filter(Dimension=dim_to_plot)
                .group_by('alt_id', 'Model', 'times')
                .agg(pl.col('State').mean().alias('Mean'))
                .collect().to_pandas()
            ),
            x='times',
            y='Mean',
            row=plot_identifier,
            row_order=row_order,
            style=plot_identifier,
            hue=plot_identifier,
            hue_order=hue_order,
            aspect=3,
        )
        # .set(ylim=(-15, 15))
        # .set(xlim=(None, 32))
    )
    map_true_state_trajectory(plot_predicted)
    sns.move_legend(
        plot_predicted,
        loc='upper center',
        ncol=min(len(alt_ids), 2) + 1,
        title='',
        bbox_to_anchor=(.455, 1.06),
        frameon=True,
        fancybox=True,
    )
    plot_predicted
    return (plot_predicted,)


@app.cell
def _(
    alt_ids,
    dim_to_plot,
    hue_order,
    map_true_state_trajectory,
    pl,
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
                .filter(Dimension=dim_to_plot)
                .collect().to_pandas()
            ),
            x='times',
            y='State',
            bins=(true_state_trajectory.select(pl.col('times')).collect().n_unique(), 101),
            row=plot_identifier,
            row_order=row_order,
            hue=plot_identifier,
            hue_order=hue_order,
            aspect=3,
            zorder=-1,
        )
        # .set(ylim=(-15, 15))
        # .set(xlim=(None, 40))
    )
    map_true_state_trajectory(plot_histogram)
    sns.move_legend(
        plot_histogram,
        loc='upper center',
        ncol=min(len(alt_ids), 2) + 1,
        title='',
        bbox_to_anchor=(.455, 1.06),
        frameon=True,
        fancybox=True,
    )
    plot_histogram
    return (plot_histogram,)


@app.cell
def _(
    alt_ids,
    dim_to_plot,
    hue_order,
    map_true_state_trajectory,
    plot_identifier,
    predicted_state_trajectory,
    row_order,
    sns,
):
    plot_confidence = (
        sns.relplot(
            kind='line',
            errorbar=('ci', 95),
            data=(
                predicted_state_trajectory
                .filter(Dimension=dim_to_plot)
                .collect().to_pandas()
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
        .set(xlim=(30, 40))
    )
    map_true_state_trajectory(plot_confidence)
    sns.move_legend(
        plot_confidence,
        loc='upper center',
        ncol=min(len(alt_ids), 2) + 1,
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
