"""Export TensorBoard scalars of an experiment (all lightning_logs/version_* folders) to one CSV.

  python3 tools/export_scalars.py experiments/<id> OUT.csv

Columns: version, tag, step, epoch, value. `epoch` comes from Lightning's own `epoch` scalar
(nearest logged step at or before each value), so resumed runs (several versions) line up.
"""
import bisect
import csv
import glob
import os

import click
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@click.command()
@click.argument('exp_dir')
@click.argument('out_csv')
def main(exp_dir, out_csv):
    rows = []
    for vdir in sorted(glob.glob(os.path.join(exp_dir, 'lightning_logs', 'version_*')),
                       key=lambda p: int(p.rsplit('_', 1)[1])):
        ea = EventAccumulator(vdir, size_guidance={'scalars': 0})
        ea.Reload()
        tags = ea.Tags()['scalars']
        epochs = sorted((e.step, e.value) for e in ea.Scalars('epoch')) if 'epoch' in tags else []
        steps = [s for s, _ in epochs]
        version = os.path.basename(vdir)
        for tag in tags:
            if tag == 'epoch':
                continue
            for e in ea.Scalars(tag):
                if tag.endswith('_step') and tag.startswith('val/'):
                    epoch = ''  # validation "step" counter, not the global step
                else:
                    i = bisect.bisect_right(steps, e.step) - 1
                    epoch = int(epochs[i][1]) if i >= 0 else ''
                rows.append({'version': version, 'tag': tag, 'step': e.step, 'epoch': epoch, 'value': e.value})
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['version', 'tag', 'step', 'epoch', 'value'])
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} scalars from {exp_dir} -> {out_csv}')


if __name__ == '__main__':
    main()
