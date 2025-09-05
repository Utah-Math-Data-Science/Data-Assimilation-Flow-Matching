import copy
from pathlib import Path

import duckdb
import polars as pl


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
            save_name = f'{plot_name}.{"__".join(variable_names)}.{format}'
        else:
            save_name = f'{plot_name}.{format}'
        pp.savefig(save_name, format=format, bbox_inches='tight', pad_inches=.06)


def get_logged_metrics_file_paths(conf_rows, file_path_format='~/out/dafm/runs/{}/dataset_metrics.csv'):
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
