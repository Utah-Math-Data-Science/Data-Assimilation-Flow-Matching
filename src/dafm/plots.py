import copy
from pathlib import Path

import duckdb
import polars as pl


FILTER_ORDER = {
    'EnSF': 0,
    'EnFF-OT': 1,
    'EnFF-F2P': 2,
    'BPF': 3,
    'EnKF-PO': 4,
    'iEnKF-PO': 5,
    'ESRF': 6,
    'LETKF': 7,
    'NoisedObs': 8,
}


FILTER_PALETTE = {
    'EnSF': 'tab:gray',
    'EnFF-OT': 'tab:orange',
    'EnFF-F2P': 'tab:red',
    'BPF': 'tab:olive',
    'EnKF-PO': 'tab:pink',
    'iEnKF-PO': 'black',
    'ESRF': 'tab:purple',
    'LETKF': 'tab:green',
    'NoisedObs': 'black',
}


def save_all_subfigures(plot, plot_name, format='pdf', renaming=None, metadata_dataframe=None):
    if metadata_dataframe is not None:
        metadata_dataframe.write_csv(f'{plot_name}.csv')
    renaming = renaming or {}
    p = copy.deepcopy(plot)
    if p._legend is not None:
        p.figure.savefig(
            f'{plot_name}.legend.{format}', format=format,
            bbox_inches=p._legend.get_window_extent().transformed(p.figure.dpi_scale_trans.inverted()).expanded(1.007, 1.1)
        )
        p._legend.set_visible(False)

    # save each subplot
    for (row, col, hue), data in p.facet_data():
        pp = copy.deepcopy(p)
        # ax = p.axes[row][col]
        for (r, c, h), d in pp.facet_data():
            if r != row or c != col:
                ax_other = pp.axes[r][c]
                ax_other.remove()
        variable_names = []
        if len(p.row_names) > 0:
            variable_names.append(renaming.get(p.row_names[row], p.row_names[row]))
        if len(p.col_names) > 0:
            variable_names.append(renaming.get(p.col_names[col], p.col_names[col]))
        if len(variable_names) > 0:
            save_name = f'{plot_name}.{"__".join(map(str, variable_names))}.{format}'
        else:
            save_name = f'{plot_name}.{format}'
        pp.savefig(save_name, format=format, bbox_inches='tight', pad_inches=.06)


def get_logged_metrics_file_paths(conf_rows, file_path_format='~/out/revision-dafm/runs/{}/metrics.csv'):
    paths = duckdb.sql(f"""
        select format({file_path_format!r}, alt_id) as path from conf_rows
    """).pl()
    exists = []
    for f in paths['path']:
        f = Path(f).expanduser()
        exists.append(f.exists())
    paths = pl.DataFrame(dict(
        path=paths['path'], exists=exists,
    ))
    return duckdb.sql('select * from paths')


def get_logged_metrics(alt_ids, file_paths=None):
    if file_paths is None:
        file_paths = get_logged_metrics_file_paths(alt_ids)
    duckdb.sql("""
    set variable dataset_metrics_filepaths = (
        select list(path) from file_paths where exists
    )
    """)
    logged_metrics = duckdb.sql("""
    select
        split(filename, '/')[-2] as alt_id,
        * exclude(filename),
    from read_csv(getvariable(dataset_metrics_filepaths), filename=true, union_by_name=true)
    """)
    return logged_metrics


def get_run_ranks(dataset_rows, filter_rows, equivalence_columns, loss_expr, max_width=120):
    rows = duckdb.sql("""
    select
        *
    from Conf
    join dataset_rows on Conf.Dataset = dataset_rows.dataset_id
    join filter_rows on Conf.Filter = filter_rows.filter_id
    join Splitter on Conf.Splitter = Splitter.id
    join StartAndLen on Splitter.id = StartAndLen.id
    where true
    and not save_ensemble_stats
    """)
    print('Rows')
    rows.show(max_width=max_width)
    logged_metrics_file_paths = get_logged_metrics_file_paths(rows)
    duckdb.sql("""
    set variable dataset_metrics_filepaths = (
        select list(path) from logged_metrics_file_paths where exists
    )
    """)
    logged_metrics = duckdb.sql("""
    select logs.*,
    from (
        select split(filename, '/')[-2] as alt_id, step, da_time_s, rmse,
        from read_csv(getvariable(dataset_metrics_filepaths), filename=true, union_by_name=true)
    ) as logs
    join rows using (alt_id)
    where true
    --and step >= ((dataset_time_step_count - start_train) / observe_every_n_time_steps)
    order by step desc
    """)
    print('Logged metrics')
    logged_metrics.show(max_width=max_width)
    rows_did_not_finish = duckdb.sql("""
    select
        alt_id,
        max(step),
    from logged_metrics
    join rows using (alt_id)
    group by alt_id, dataset_time_step_count, start_train, observe_every_n_time_steps
    having max(step) < ((dataset_time_step_count - start_train) / observe_every_n_time_steps) - 1
    """)
    print('Rows that did not finish')
    rows_did_not_finish.show(max_width=max_width)
    logged_metrics = duckdb.sql("""
    select
        *
    from logged_metrics
    where alt_id not in (select alt_id from rows_did_not_finish)
    """).pl()
    equivalence_class = ', '.join(equivalence_columns)
    ranked_rows = duckdb.sql(f"""
    with mean_loss_per_run as (
        select
            alt_id,
            {loss_expr} as loss,
        from logged_metrics
        group by alt_id
    ),
    ranks as (
        select
            rows.alt_id,
            {equivalence_class},
            mean_loss_per_run.loss,
            dense_rank() over (partition by {equivalence_class} order by loss) as top,
        from mean_loss_per_run
        join rows using (alt_id)
    )
    select
        alt_id,
        {equivalence_class},
        loss,
        top,
        row_number() over (partition by {equivalence_class}, top order by top) as tie_breaker,
    from ranks
    order by {equivalence_class}, top, tie_breaker
    """)
    return ranked_rows
