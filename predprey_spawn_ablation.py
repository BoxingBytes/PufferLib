"""Spawn-distance ablation for the PredPrey fireplace task.

Trains one policy per spawn distance (0..MAX_DISTANCE) and records the best score
obtained, then plots best score vs. spawn distance.

Safety nets:
  * Results are written to the JSON file after EVERY run, so a cancelled run only
    loses the run currently in progress.
  * On startup, distances already present in the JSON file are skipped, so you can
    resume an interrupted ablation without redoing finished runs.

Usage:
    python predprey_spawn_ablation.py
    python predprey_spawn_ablation.py --max-distance 4 --total-timesteps 5_000_000
    python predprey_spawn_ablation.py --force          # ignore existing results
"""
import argparse
import json
import os
import sys

ENV_NAME = 'puffer_predprey'
SCORE_KEY = 'environment/score'


def best_score_from_logs(all_logs):
    """Best (max) environment/score across every logged point of a run."""
    scores = [log[SCORE_KEY] for log in all_logs
              if log is not None and SCORE_KEY in log]
    if not scores:
        return None, []
    return float(max(scores)), [float(s) for s in scores]


def load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {'env_name': ENV_NAME, 'score_key': SCORE_KEY, 'runs': {}}


def save_results(path, results):
    # Write to a temp file then rename, so an interrupt mid-write can't corrupt it.
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, path)


def run_one(distance, total_timesteps):
    """Train a single policy at the given spawn distance, return (best, all_scores)."""
    # Imported lazily so --help is fast and torch isn't imported needlessly.
    from pufferlib import pufferl

    args = pufferl.load_config(ENV_NAME)
    args['env']['spawn_distance'] = distance
    if total_timesteps is not None:
        args['train']['total_timesteps'] = int(total_timesteps)
    # Keep each run self-contained and avoid clobbering a real wandb project.
    args['wandb'] = False
    args['neptune'] = False

    all_logs = pufferl.train(ENV_NAME, args=args)
    return best_score_from_logs(all_logs)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--max-distance', type=int, default=4,
                        help='Run distances 0..max-distance inclusive (default 4 -> 5 runs).')
    parser.add_argument('--total-timesteps', type=str, default=None,
                        help='Override train.total_timesteps for each run (default: config value).')
    parser.add_argument('--out', type=str, default='predprey_spawn_ablation.json',
                        help='JSON file to read/write results.')
    parser.add_argument('--plot', type=str, default='predprey_spawn_ablation.png',
                        help='Output path for the best-score-vs-distance plot.')
    parser.add_argument('--force', action='store_true',
                        help='Re-run all distances, ignoring any already in the JSON file.')
    cli = parser.parse_args()

    # pufferl.load_config() runs its own argparse over sys.argv; clear our args so
    # it falls back to the config-file defaults instead of choking on them.
    sys.argv = [sys.argv[0]]

    total_timesteps = None
    if cli.total_timesteps is not None:
        total_timesteps = int(cli.total_timesteps.replace('_', ''))

    results = {'env_name': ENV_NAME, 'score_key': SCORE_KEY, 'runs': {}} if cli.force \
        else load_results(cli.out)

    for distance in range(cli.max_distance + 1):
        key = str(distance)
        if not cli.force and key in results['runs']:
            print(f'[ablation] spawn_distance={distance} already done '
                  f"(best={results['runs'][key]['best_score']}); skipping.")
            continue

        print(f'\n[ablation] ===== Training spawn_distance={distance} =====')
        best, all_scores = run_one(distance, total_timesteps)
        results['runs'][key] = {
            'spawn_distance': distance,
            'best_score': best,
            'all_scores': all_scores,
            'total_timesteps': total_timesteps,
        }
        save_results(cli.out, results)  # <-- safety net: persist after each run
        print(f'[ablation] spawn_distance={distance} done. best_score={best}. '
              f'Saved to {cli.out}.')

    make_plot(results, cli.plot)


def make_plot(results, plot_path):
    runs = results['runs']
    distances = sorted(int(k) for k in runs)
    xs = [d for d in distances if runs[str(d)]['best_score'] is not None]
    ys = [runs[str(d)]['best_score'] for d in xs]
    if not xs:
        print('[ablation] No scores to plot.')
        return

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[ablation] matplotlib not installed; skipping plot. Data is in the JSON.')
        return

    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys, marker='o')
    plt.xlabel('Spawn distance to fireplace (steps)')
    plt.ylabel('Best score obtained')
    plt.title('PredPrey: best score vs. spawn distance to fireplace')
    plt.xticks(xs)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    print(f'[ablation] Plot saved to {plot_path}.')


if __name__ == '__main__':
    main()
