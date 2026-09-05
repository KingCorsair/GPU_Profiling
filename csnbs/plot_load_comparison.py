"""Plot saved VisPruner measurements with Matplotlib; never run a benchmark.

The four-condition figure uses matching 576-token controls. Missing isolated or
engineered runs remain unmeasured. The supporting figure shows all three saved
latency percentiles and successful throughput against offered open-loop load.
See plotting.md for the campaign format and commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = REPO_ROOT / 'results/loadgen/2026-09-01/vispruner_stock_ab_campaign.json'
DEFAULT_OUTPUT = REPO_ROOT / 'results/loadgen/vispruner_under_load'
PERCENTILES = ('p50', 'p95', 'p99')
COLORS = ('#2563a6', '#de7625', '#168776', '#a84793')
MARKERS = ('o', 's', '^', 'D')
STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelcolor': '#273343', 'text.color': '#273343',
    'axes.titleweight': 'bold', 'axes.axisbelow': True,
    'grid.color': '#e0e5eb', 'grid.linewidth': 0.8,
    'svg.fonttype': 'none', 'svg.hashsalt': 'vispruner-load-comparison',
}
FIELD_MAP = {
    'offeredRps': ('config', 'requestsPerSecond'),
    'measuredRequests': ('summary', 'totalRequests'),
    'successfulRequests': ('summary', 'successfulRequests'),
    'failedRequests': ('summary', 'failedRequests'),
    'successfulThroughputRps': ('summary', 'successfulThroughputRps'),
    **{f'{p}LatencyMs': ('summary', 'successfulRequestLatencyMs', p) for p in PERCENTILES},
}


def get_value(data: dict, path: tuple[str, ...]):
    for key in path:
        data = data[key]
    return data


def number(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite number')
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return float(value)


def load_condition(item: dict) -> str:
    return item.get('loadCondition', 'under-load')


def hydrate_and_verify_raw_runs(data: dict, repo_root: Path = REPO_ROOT) -> None:
    """Verify cached values, then use the harness summaries as the only data source."""
    if data.get('schema') != 'vispruner-load-campaign-v1':
        raise ValueError('expected schema vispruner-load-campaign-v1')
    campaign = data['campaign']
    if not data.get('series'):
        raise ValueError('campaign must contain measured series')
    seen_ids, seen_roles, seen_runs = set(), set(), set()
    for item in data['series']:
        series_id = item['id']
        condition = load_condition(item)
        role = (item['servingPath'], item['visualTokenNum'], condition)
        if not isinstance(series_id, str) or not series_id or series_id in seen_ids:
            raise ValueError('series must have unique nonempty string IDs')
        if role in seen_roles:
            raise ValueError(f'duplicate serving path/token count/load condition: {role}')
        seen_ids.add(series_id)
        seen_roles.add(role)
        if item['servingPath'] not in ('stock', 'engineered'):
            raise ValueError('servingPath must be stock or engineered')
        tokens = number(item['visualTokenNum'], 'visualTokenNum', positive=True)
        if not tokens.is_integer() or tokens > 576:
            raise ValueError('visualTokenNum must be an integer from 1 through 576')
        if condition not in ('under-load', 'isolated'):
            raise ValueError('loadCondition must be under-load or isolated')
        runs = item['runs']
        if not runs or (condition == 'isolated' and len(runs) != 1):
            raise ValueError('series need runs; isolated series need exactly one saved summary')
        seen_rates = set()
        for run in runs:
            run_path = Path(run['runPath'])
            if not run_path.is_absolute():
                run_path = repo_root / run_path
            raw_bytes = run_path.read_bytes()
            raw = json.loads(raw_bytes)
            if raw['runId'] != run['runId'] or raw['runId'] in seen_runs:
                raise ValueError(f'{run_path}: mismatched or duplicate runId')
            seen_runs.add(raw['runId'])
            expected_kind = 'open-loop' if condition == 'under-load' else 'isolated'
            if raw.get('runKind') != expected_kind:
                raise ValueError(f'{run_path}: {condition} requires runKind={expected_kind}')
            if raw.get('schema') != 'loadgen-run' or raw.get('schemaVersion') != 1:
                raise ValueError(f'{run_path}: unsupported harness summary schema')
            for field, path in FIELD_MAP.items():
                if condition == 'isolated' and field == 'offeredRps':
                    continue
                value = get_value(raw, path)
                number(value, f'{run_path}: {field}', positive=field in ('offeredRps', 'measuredRequests'))
                if field in run and not math.isclose(
                    number(run[field], field), value, rel_tol=1e-12, abs_tol=1e-9
                ):
                    raise ValueError(f'{run["runId"]}: {field} does not match {run_path}')
                run[field] = value
            for field in ('measuredRequests', 'successfulRequests', 'failedRequests'):
                if not float(run[field]).is_integer():
                    raise ValueError(f'{run_path}: {field} must be an integer')
            if run['successfulRequests'] + run['failedRequests'] != run['measuredRequests']:
                raise ValueError(f'{run_path}: success/failure counts do not sum to measured count')
            if run['successfulRequests'] == 0:
                raise ValueError(f'{run_path}: no successful requests to plot latency')
            recorded_count = raw.get('requests', {}).get('count')
            if recorded_count is not None:
                excluded = recorded_count - run['measuredRequests']
                if excluded != campaign['warmupRequestsExcludedPerLevel']:
                    raise ValueError(f'{run_path}: excluded request count contradicts campaign warmups')
            if not 0 < run['p50LatencyMs'] <= run['p95LatencyMs'] <= run['p99LatencyMs']:
                raise ValueError(f'{run_path}: latency percentiles must be positive and ordered')
            if condition == 'under-load':
                if run['offeredRps'] in seen_rates:
                    raise ValueError(f'{series_id}: duplicate offered-RPS level')
                seen_rates.add(run['offeredRps'])
            # These fields exist in the saved runs, unlike some model/pruning metadata.
            expected = [
                (('workload', 'id'), campaign['workloadId']),
                (('hardware', 'gpuModels'), [campaign['gpu']]),
                (('source', 'gitCommit'), item.get('gitCommit', campaign['gitCommit'])),
                (('benchmark', 'batchSize'), item.get('batchSize', campaign['batchSize'])),
            ]
            if condition == 'under-load':
                expected.append((('config', 'durationSeconds'), campaign['durationSecondsPerLevel']))
            for path, value in expected:
                if get_value(raw, path) != value:
                    raise ValueError(f'{run_path}: {".".join(path)} does not match campaign/series')
            for field, expected_value in (
                ('maxOutputTokens', campaign['maxNewTokens']),
                ('visualTokenNum', item['visualTokenNum']),
                ('importantRatio', campaign['importantRatio']),
            ):
                actual = raw['benchmark'].get(field)
                if actual is not None and actual != expected_value:
                    raise ValueError(f'{run_path}: benchmark.{field} contradicts campaign/series')
            run['_raw'] = raw
            run['_sha256'] = hashlib.sha256(raw_bytes).hexdigest()
        if condition == 'under-load':
            runs.sort(key=lambda run: run['offeredRps'])


def find_run(data: dict, path: str, tokens: int, condition: str, rps: float):
    for item in data['series']:
        if (item['servingPath'], item['visualTokenNum'], load_condition(item)) != (path, tokens, condition):
            continue
        for run in item['runs']:
            if condition == 'isolated' or math.isclose(run['offeredRps'], rps, rel_tol=1e-12):
                return run
    return None


def verify_pair(control: dict, pruned: dict) -> None:
    """Reject known mismatches; do not infer missing model settings."""
    a, b = control['_raw'], pruned['_raw']
    for path in (
        ('workload',), ('hardware', 'gpuModels'), ('source', 'gitCommit'),
        ('config', 'durationSeconds'), ('config', 'timeoutMs'),
        ('benchmark', 'batchSize'), ('benchmark', 'modelId'),
        ('benchmark', 'checkpoint'), ('benchmark', 'maxOutputTokens'),
        ('benchmark', 'importantRatio'),
    ):
        # Optional metadata may be null in historical harness summaries.
        left, right = a.get(path[0], {}), b.get(path[0], {})
        for key in path[1:]:
            left = left.get(key) if isinstance(left, dict) else None
            right = right.get(key) if isinstance(right, dict) else None
        if left != right:
            raise ValueError(f'unmatched controls: {".".join(path)} differs')
    if control['measuredRequests'] != pruned['measuredRequests']:
        raise ValueError('unmatched controls: measured request counts differ')


def story_rows(data: dict, *, tokens: int, rps: float, percentile: str) -> list[dict]:
    rows = [{'label': 'No pruning\n576 tokens', 'improvementPercent': 0.0,
             'note': 'Reference by definition', 'controlRunId': None, 'prunedRunId': None}]
    for label, path, condition in (
        ('Stock VisPruner\nNo load', 'stock', 'isolated'),
        (f'Stock VisPruner\nUnder load · {rps:g} RPS', 'stock', 'under-load'),
        (f'Optimized VisPruner\nUnder load · {rps:g} RPS', 'engineered', 'under-load'),
    ):
        control = find_run(data, path, 576, condition, rps)
        pruned = find_run(data, path, tokens, condition, rps)
        row = {'label': label, 'improvementPercent': None, 'note': 'Run required',
               'controlRunId': control['runId'] if control else None,
               'prunedRunId': pruned['runId'] if pruned else None}
        if control and pruned:
            verify_pair(control, pruned)
            field = f'{percentile}LatencyMs'
            row.update(
                improvementPercent=100 * (control[field] - pruned[field]) / control[field],
                controlLatencyMs=control[field], prunedLatencyMs=pruned[field],
                note=f'n={control["successfulRequests"]}/{pruned["successfulRequests"]} successful (control/pruned)',
            )
        rows.append(row)
    return rows


def provenance_lines(data: dict) -> list[str]:
    campaign = data['campaign']
    runs = [run for item in data['series'] for run in item['runs']]
    counts = [run['successfulRequests'] for run in runs]
    failed = sum(run['failedRequests'] for run in runs)
    commits = sorted({run['_raw']['source']['gitCommit'][:12] for run in runs})
    dirty = any(run['_raw']['source']['gitDirty'] for run in runs)
    incomplete = any(any(run['_raw']['benchmark'].get(key) is None for key in (
        'modelId', 'checkpoint', 'maxOutputTokens', 'visualTokenNum', 'importantRatio'
    )) for run in runs)
    lines = [
        f'{campaign["gpu"]} | commit {", ".join(commits)}{" (dirty worktree)" if dirty else ""} | '
        f'workload {campaign["workloadId"][:23]}…',
        f'Exploratory: one run per point; n={min(counts)}–{max(counts)} successful requests; '
        f'{campaign["warmupRequestsExcludedPerLevel"]} warmups excluded per campaign. No confidence intervals.',
    ]
    if min(counts) < 1000:
        lines.append('p99 has fewer than 1,000 successful samples at one or more points; tail estimates are unstable.')
    if incomplete:
        lines.append('Some model/pruning settings exist only in the campaign; raw-run metadata cannot fully verify matching.')
    if failed:
        lines.append(f'{failed} failed requests across these runs; latency percentiles describe successful requests only.')
    if campaign.get('serverConcurrency'):
        lines.append(f'Serving (campaign): {campaign["serverConcurrency"]}.')
    return lines


def add_footer(fig, lines: list[str]) -> None:
    wrapped = [textwrap.fill(line, width=150) for line in lines]
    fig.text(0.07, 0.025, '\n'.join(wrapped), fontsize=9, color='#596777', va='bottom', linespacing=1.6)


def build_story(data: dict, rows: list[dict], *, tokens: int, percentile: str):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.subplots_adjust(left=0.095, right=0.97, top=0.80, bottom=0.35)
    fig.suptitle('Does visual-token pruning keep its speedup under load?', fontsize=19, fontweight='bold', y=0.955)
    fig.text(0.5, 0.898, f'{tokens} retained tokens · {percentile} latency reduction against each matching 576-token control',
             ha='center', fontsize=12)
    values = [row['improvementPercent'] for row in rows if row['improvementPercent'] is not None]
    extent = max(10, *(abs(value) for value in values)) * 1.55
    ax.set_ylim(-extent, extent)
    ax.set_xlim(-0.6, 3.6)
    ax.axhline(0, color='#35465b', linewidth=1.2)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_ylabel(f'{percentile} latency reduction (%)')
    ax.grid(axis='y')
    ax.set_xticks(range(4), [row['label'] for row in rows])
    ax.tick_params(axis='x', length=0, pad=15)
    for index, row in enumerate(rows):
        value = row['improvementPercent']
        if value is None:
            # No bar and no numeric height: missing data must not look like zero improvement.
            ax.axvspan(index - 0.34, index + 0.34, color='#edf0f4', zorder=0)
            ax.text(index, 0, 'Run required', ha='center', va='center', color='#596777',
                    bbox={'facecolor': 'white', 'edgecolor': '#bcc5d0', 'pad': 9})
        elif index == 0:
            ax.plot(index, 0, marker='_', markersize=65, markeredgewidth=3, color=COLORS[0])
            ax.annotate('0% reference', (index, 0), xytext=(0, 12), textcoords='offset points', ha='center')
        else:
            ax.bar(index, value, width=0.60, color='#168776' if value >= 0 else '#c94b48', zorder=3)
            ax.annotate(f'{value:+.1f}%', (index, value), xytext=(0, 9 if value >= 0 else -9),
                        textcoords='offset points', ha='center', va='bottom' if value >= 0 else 'top', fontweight='bold')
        ax.text(index, -0.26, row['note'].replace(' successful (control/pruned)', '\ncontrol/pruned successes'),
                transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=9, color='#596777')
    ax.text(0.99, 1.03, 'Positive = faster   |   Negative = slower', transform=ax.transAxes,
            ha='right', fontsize=10, color='#596777')
    add_footer(fig, provenance_lines(data))
    return fig


def build_evidence(data: dict):
    import matplotlib.pyplot as plt

    series = [item for item in data['series'] if load_condition(item) == 'under-load']
    if not series:
        raise ValueError('at least one under-load series is needed for the evidence chart')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    fig.subplots_adjust(left=0.075, right=0.96, top=0.82, bottom=0.24, wspace=0.20, hspace=0.40)
    fig.suptitle('VisPruner under open-loop load', fontsize=21, fontweight='bold', y=0.967)
    fig.text(0.5, 0.923, 'Saved request-latency percentiles and successful throughput · exploratory measurements',
             ha='center', fontsize=12)
    flat_axes = list(axes.flat)
    for ax, percentile in zip(flat_axes[:3], PERCENTILES):
        ax.set_title(f'{percentile} request latency', loc='left', pad=12)
        ax.set_ylabel('Latency (seconds)')
    flat_axes[3].set_title('Successful throughput', loc='left', pad=12)
    flat_axes[3].set_ylabel('Successful requests / second')
    rates = sorted({run['offeredRps'] for item in series for run in item['runs']})
    for index, item in enumerate(series):
        runs = item['runs']
        style = {'color': COLORS[index % 4], 'marker': MARKERS[index % 4], 'linewidth': 2,
                 'markersize': 7, 'markeredgecolor': 'white', 'markeredgewidth': 0.8, 'label': item['label']}
        xs = [run['offeredRps'] for run in runs]
        for ax, percentile in zip(flat_axes[:3], PERCENTILES):
            ax.plot(xs, [run[f'{percentile}LatencyMs'] / 1000 for run in runs], **style)
        flat_axes[3].plot(xs, [run['successfulThroughputRps'] for run in runs], **style)
    flat_axes[3].plot([0, max(rates)], [0, max(rates)], '--', color='#8994a3', linewidth=1.2, label='Offered rate')
    for ax in flat_axes:
        ax.set_xlabel('Offered load (requests / second)')
        ax.set_xticks(rates)
        ax.set_xlim(max(0, min(rates) - 0.12), max(rates) + 0.12)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y')
    handles, labels = flat_axes[3].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.897),
               ncol=min(len(labels), 3), frameon=False)
    add_footer(fig, provenance_lines(data))
    return fig


def save_figures(data: dict, rows: list[dict], output: Path, *, tokens: int, percentile: str, dpi: int) -> list[Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    saved = []
    with plt.rc_context(STYLE):
        for suffix, build in (
            ('', lambda: build_evidence(data)),
            ('_story', lambda: build_story(data, rows, tokens=tokens, percentile=percentile)),
        ):
            fig = build()
            try:
                for extension in ('png', 'svg'):
                    path = output.parent / f'{output.name}{suffix}.{extension}'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(path, dpi=dpi, facecolor='white', metadata={
                        'Description': '\n'.join(provenance_lines(data)),
                    })
                    saved.append(path)
            finally:
                plt.close(fig)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', nargs='?', type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument('-o', '--output', type=Path, default=DEFAULT_OUTPUT,
                        help='output stem (a .png/.svg suffix is also accepted); writes both formats')
    parser.add_argument('--percentile', choices=PERCENTILES, default='p50', help='story percentile; evidence always includes all three')
    parser.add_argument('--rps', type=float, default=3.0, help='offered load selected for the story (default: 3)')
    parser.add_argument('--tokens', type=int, default=128, help='retained token count for the story (default: 128)')
    parser.add_argument('--dpi', type=int, default=180)
    args = parser.parse_args()
    try:
        number(args.rps, '--rps', positive=True)
        if not 1 <= args.tokens < 576 or args.dpi <= 0:
            raise ValueError('--tokens must be 1–575 and --dpi must be positive')
        campaign_bytes = args.campaign.read_bytes()
        data = json.loads(campaign_bytes)
        hydrate_and_verify_raw_runs(data)
        # Validate all available control/pruned pairs, including non-story load levels.
        for item in data['series']:
            if item['visualTokenNum'] == 576:
                continue
            for run in item['runs']:
                control = find_run(data, item['servingPath'], 576, load_condition(item), run.get('offeredRps', 0))
                if control:
                    verify_pair(control, run)
        rows = story_rows(data, tokens=args.tokens, rps=args.rps, percentile=args.percentile)
        output = args.output.with_suffix('') if args.output.suffix in ('.png', '.svg') else args.output
        paths = save_figures(data, rows, output, tokens=args.tokens, percentile=args.percentile, dpi=args.dpi)
        report = {
            'campaign': data['campaign'],
            'campaignPath': str(args.campaign.resolve()),
            'campaignSha256': hashlib.sha256(campaign_bytes).hexdigest(),
            'percentile': args.percentile, 'offeredRps': args.rps, 'visualTokenNum': args.tokens,
            'formula': '100 * (control_ms - pruned_ms) / control_ms',
            'conditions': rows, 'notes': provenance_lines(data),
            'series': [{**item, 'runs': [{
                **{key: value for key, value in run.items() if not key.startswith('_')},
                'sha256': run['_sha256'], 'source': run['_raw']['source'],
                'hardware': run['_raw']['hardware'], 'benchmark': run['_raw']['benchmark'],
            } for run in item['runs']]} for item in data['series']],
        }
        report_path = output.parent / f'{output.name}_sources.json'
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        for path in [*paths, report_path]:
            print(f'Saved {path}')
        for row in rows:
            value = row['improvementPercent']
            print(f'{row["label"].replace(chr(10), " / ")}: {value:+.2f}%' if value is not None
                  else f'{row["label"].replace(chr(10), " / ")}: Run required')
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    except ModuleNotFoundError as error:
        parser.exit(2, f'{error}. Install the plotting environment described in csnbs/plotting.md.\n')


if __name__ == '__main__':
    main()
