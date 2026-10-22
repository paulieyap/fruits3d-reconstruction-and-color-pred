"""Tables and error-vs-occlusion plots from baselines/evaluate.py output.

  python3 -m baselines.report RESULTS_CSV OUT_DIR [--exclude p26,p28] [--methods ddpm,sphere,...]

Writes OUT_DIR/tables.md (overall / lab / greenhouse / greenhouse without excluded fruits) and
OUT_DIR/volerr_vs_visible.png, OUT_DIR/fscore_vs_visible.png (one panel per method).
"""
import csv
import os
import statistics as st

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# validated categorical slots 1-2 (dataviz reference palette), plus marker shape as second encoding
SUBSET_STYLE = {'lab': dict(color='#2a78d6', marker='o'), 'greenhouse': dict(color='#eb6834', marker='^')}
INK, MUTED, GRID = '#2b2b29', '#6b6a64', '#e4e3df'

COLUMNS = [('cd_mm', 'CD [mm]', 'mean'), ('fscore', 'F [%]', 'mean'), ('precision', 'P [%]', 'mean'),
           ('recall', 'R [%]', 'mean'), ('fscore_5mm', 'F@5mm [%]', 'mean'),
           ('vol_err_pct', 'median vol. err [%]', 'median'), ('abs_vol_err_pct', 'median abs. vol. err [%]', 'median'),
           ('abs_vol_err_pct', 'mean abs. vol. err [%]', 'mean')]


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = float(v)
            except ValueError:
                pass
        r['abs_vol_err_pct'] = abs(r['vol_err_pct'])
    return rows


def table(rows, methods, title):
    lines = [f'**{title}** (n = {len({r["fruit_id"] for r in rows})} fruits)', '',
             '| method | ' + ' | '.join(c[1] for c in COLUMNS) + ' |',
             '|---|' + '---:|' * len(COLUMNS)]
    for m in methods:
        rs = [r for r in rows if r['method'] == m]
        if not rs:
            continue
        agg = {'mean': st.mean, 'median': st.median}
        lines.append(f'| {m} | ' + ' | '.join(f'{agg[a]([r[k] for r in rs]):.2f}' for k, _, a in COLUMNS) + ' |')
    return '\n'.join(lines) + '\n'


def scatter(rows, methods, key, ylabel, path, exclude, log=False):
    n = len(methods)
    cols = min(5, n)
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(3.2 * cols, 3.0 * nrows), sharex=True, sharey=True, squeeze=False)
    for ax, m in zip(axes.flat, methods):
        rs = [r for r in rows if r['method'] == m]
        for subset, style in SUBSET_STYLE.items():
            for excluded in (False, True):
                pts = [r for r in rs if r['subset'] == subset and (r['fruit_id'] in exclude) == excluded]
                if not pts:
                    continue
                ax.scatter([r['visible_frac'] for r in pts], [r[key] for r in pts], s=28, marker=style['marker'],
                           facecolors='none' if excluded else style['color'], edgecolors=style['color'],
                           linewidths=1.2, label=f'{subset}{" (misaligned GT)" if excluded else ""}')
        ax.set_title(m, color=INK, fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        if log:
            ax.set_yscale('log')
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('GT surface seen in partial', color=INK, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=110, facecolor='white')
    plt.close(fig)


@click.command()
@click.argument('results_csv')
@click.argument('out_dir')
@click.option('--exclude', default='p26,p28', help='fruits with misaligned GT, reported separately')
@click.option('--methods', default=None, help='comma-separated methods and order (default: all, as in the csv)')
def main(results_csv, out_dir, exclude, methods):
    os.makedirs(out_dir, exist_ok=True)
    rows = load(results_csv)
    exclude = set(e for e in exclude.split(',') if e)
    methods = methods.split(',') if methods else list(dict.fromkeys(r['method'] for r in rows))

    parts = [table(rows, methods, 'All val fruits'),
             table([r for r in rows if r['subset'] == 'lab'], methods, 'Lab'),
             table([r for r in rows if r['subset'] == 'greenhouse'], methods, 'Greenhouse'),
             table([r for r in rows if r['subset'] == 'greenhouse' and r['fruit_id'] not in exclude], methods,
                   f'Greenhouse without {", ".join(sorted(exclude))} (misaligned GT)')]
    with open(os.path.join(out_dir, 'tables.md'), 'w') as f:
        f.write('\n'.join(parts))

    scatter(rows, methods, 'abs_vol_err_pct', '|volume error| [%]', os.path.join(out_dir, 'volerr_vs_visible.png'),
            exclude, log=True)
    scatter(rows, methods, 'fscore', 'F-score (AUC 1-10 mm) [%]', os.path.join(out_dir, 'fscore_vs_visible.png'),
            exclude)
    print('\n'.join(parts))


if __name__ == '__main__':
    main()
