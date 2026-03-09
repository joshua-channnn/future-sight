import os
import sys
import time
import numpy as np
from collections import deque
from datetime import datetime

try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False
    print("Warning: tensorboard not found. Install with: pip install tensorboard")


class TrainingMonitor:
    """Monitor PPO training and detect issues early."""

    METRIC_RANGES = {
        'train/clip_fraction': {
            # clip_range=0.08 is tighter than default 0.2
            # Tighter clip → more clipping is expected
            # With clip_range=0.08, 15-30% clipping is normal
            'healthy': (0.05, 0.30),
            'warning': (0.01, 0.40),
            'description': 'Fraction of clipped updates (higher expected with clip_range=0.08)',
        },
        'train/approx_kl': {
            # With n_epochs=7 and tight clipping, KL stays moderate
            'healthy': (0.002, 0.025),
            'warning': (0.0005, 0.04),
            'description': 'KL divergence between old and new policy',
        },
        'train/entropy_loss': {
            # 13 actions → max entropy = ln(13) ≈ 2.56
            # Healthy exploration ≈ spreading across 4-6 actions
            # entropy ~1.2-1.8 → entropy_loss = -1.8 to -1.2
            # Early training should be higher (more exploration)
            # Late training will decrease as policy sharpens
            'healthy': (-2.0, -0.5),
            'warning': (-2.5, -0.2),
            'description': 'Policy entropy (13 actions: max=2.56, -1.5 ≈ 5 actions)',
            'notes': {
                'max_entropy': 2.565,  # ln(13)
                'uniform_5_actions': -1.609,  # -ln(5)
                'uniform_3_actions': -1.099,  # -ln(3)
            },
        },
        'train/explained_variance': {
            # gamma=0.9999 means returns are almost undiscounted
            # Value function has a harder job predicting long-horizon returns
            # Expect lower explained variance than with gamma=0.99
            'healthy': (0.15, 1.0),
            'warning': (0.0, 1.0),
            'description': 'Value function quality (lower expected with gamma=0.9999)',
        },
        'train/policy_gradient_loss': {
            'healthy': (-0.05, 0.0),
            'warning': (-0.1, 0.01),
            'description': 'Policy gradient loss',
        },
        'train/value_loss': {
            # With gamma=0.9999, returns have higher variance
            # Value loss will be higher than with gamma=0.99
            'healthy': (0.0, 1.0),
            'warning': (0.0, 2.0),
            'description': 'Value function loss (higher expected with gamma=0.9999)',
        },
        'train/learning_rate': {
            # Wang schedule: 5e-4 → 1e-5 with aggressive 1/(8x+1)^1.5 decay
            # Drops fast: at 10% progress already ~1.5e-4
            # At 50% progress ~3e-5
            'healthy': (5e-6, 6e-4),
            'warning': (1e-6, 1e-3),
            'description': 'Learning rate (Wang decay: 5e-4 → 1e-5)',
        },

        'eval/win_rate': {
            'healthy': (0.3, 1.0),
            'warning': (0.15, 1.0),
            'description': 'Win rate vs SimpleHeuristics',
        },

        'embeddings/pokemon_norm': {
            'healthy': (1.0, 60.0),
            'warning': (0.1, 120.0),
            'description': 'Pokemon embedding weight norm (48-dim, 508 vocab)',
        },
        'embeddings/move_norm': {
            'healthy': (1.0, 50.0),
            'warning': (0.1, 100.0),
            'description': 'Move embedding weight norm (24-dim, 351 vocab)',
        },
        'embeddings/ability_norm': {
            'healthy': (1.0, 40.0),
            'warning': (0.1, 80.0),
            'description': 'Ability embedding weight norm (16-dim, 204 vocab)',
        },
        'embeddings/item_norm': {
            'healthy': (1.0, 30.0),
            'warning': (0.1, 60.0),
            'description': 'Item embedding weight norm (16-dim, 63 vocab)',
        },
        'embeddings/type_norm': {
            'healthy': (1.0, 30.0),
            'warning': (0.1, 60.0),
            'description': 'Type embedding weight norm (16-dim, 19 vocab)',
        },

        'train/win_rate': {
            'healthy': (0.2, 0.9),
            'warning': (0.05, 0.95),
            'description': 'Win rate during training (not eval)',
        },

        'selfplay/heuristic_prob': {
            'healthy': (0.0, 1.0),
            'warning': (0.0, 1.0),
            'description': 'Heuristic opponent probability',
        },
        'selfplay/pool_size': {
            'healthy': (1, 15),
            'warning': (0, 20),
            'description': 'Self-play checkpoint pool size',
        },
    }

    MILESTONES = {
        500_000: {
            'eval/win_rate': 0.15,
            'description': 'Embeddings warming up, basic move selection',
        },
        1_000_000: {
            'eval/win_rate': 0.25,
            'description': 'Learning type matchups and basic switching',
        },
        3_000_000: {
            'eval/win_rate': 0.40,
            'description': 'Embeddings encoding useful Pokemon info',
        },
        5_000_000: {
            'eval/win_rate': 0.50,
            'description': 'Competitive play, should understand matchups',
        },
        10_000_000: {
            'eval/win_rate': 0.60,
            'description': 'Approaching v13 level, embeddings mature',
        },
        15_000_000: {
            'eval/win_rate': 0.68,
            'description': 'Should match v13 baseline (72%)',
        },
        20_000_000: {
            'eval/win_rate': 0.72,
            'description': 'Exceeding v13, embedding advantage kicks in',
        },
        30_000_000: {
            'eval/win_rate': 0.78,
            'description': 'Target: surpass v13 ceiling',
        },
    }

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        self.metrics_history = {}
        self.issues_detected = []
        self.last_step = 0

    def load_tensorboard_logs(self, run_name: str = None):
        """Load metrics from TensorBoard logs."""
        if not HAS_TB:
            return {}

        if not os.path.exists(self.log_dir):
            return {}

        # Find run directories
        run_dirs = []
        for item in os.listdir(self.log_dir):
            item_path = os.path.join(self.log_dir, item)
            if os.path.isdir(item_path):
                if run_name and run_name not in item:
                    continue
                run_dirs.append(item_path)

        if not run_dirs:
            return {}

        latest_run = max(run_dirs, key=os.path.getmtime)

        try:
            ea = event_accumulator.EventAccumulator(latest_run)
            ea.Reload()

            metrics = {}
            for tag in ea.Tags().get('scalars', []):
                events = ea.Scalars(tag)
                metrics[tag] = [(e.step, e.value) for e in events]

            return metrics
        except Exception as e:
            print(f"Error loading TensorBoard logs: {e}")
            return {}

    def analyze_metric(self, name: str, values: list) -> dict:
        """Analyze a single metric for issues."""
        if not values or name not in self.METRIC_RANGES:
            return {'status': 'unknown', 'message': 'No data'}

        ranges = self.METRIC_RANGES[name]
        recent_values = [v for _, v in values[-10:]]

        if not recent_values:
            return {'status': 'unknown', 'message': 'No recent data'}

        avg = np.mean(recent_values)
        std = np.std(recent_values) if len(recent_values) > 1 else 0
        trend = self._compute_trend(values)

        healthy_low, healthy_high = ranges['healthy']
        warning_low, warning_high = ranges['warning']

        if healthy_low <= avg <= healthy_high:
            status = 'healthy'
            symbol = '✓'
        elif warning_low <= avg <= warning_high:
            status = 'warning'
            symbol = '⚠'
        else:
            status = 'critical'
            symbol = '✗'

        # Format the value nicely
        if abs(avg) < 0.001:
            val_str = f"{avg:.6f}"
        elif abs(avg) < 1:
            val_str = f"{avg:.4f}"
        elif abs(avg) < 100:
            val_str = f"{avg:.2f}"
        else:
            val_str = f"{avg:.1f}"

        # Add trend arrow
        trend_str = ""
        if trend is not None:
            if trend > 0.05:
                trend_str = " ↑"
            elif trend < -0.05:
                trend_str = " ↓"
            else:
                trend_str = " →"

        message = f"{symbol} {val_str}{trend_str}  (healthy: {healthy_low}–{healthy_high})"

        return {
            'status': status,
            'message': message,
            'avg': avg,
            'std': std,
            'trend': trend,
            'description': ranges['description'],
        }

    def _compute_trend(self, values: list, window: int = 5) -> float:
        """Compute trend as normalized slope over recent window."""
        if len(values) < window * 2:
            return None

        early = [v for _, v in values[-(window * 2):-window]]
        recent = [v for _, v in values[-window:]]

        if not early or not recent:
            return None

        early_avg = np.mean(early)
        recent_avg = np.mean(recent)

        if abs(early_avg) < 1e-8:
            return 0.0

        return (recent_avg - early_avg) / abs(early_avg)

    def check_milestones(self, metrics: dict) -> list:
        """Check if training is meeting expected milestones."""
        issues = []

        if 'eval/win_rate' not in metrics:
            return issues

        win_rates = metrics['eval/win_rate']
        if not win_rates:
            return issues

        current_step = win_rates[-1][0]
        current_wr = win_rates[-1][1]

        # Find the highest milestone we've passed
        relevant_milestone = None
        for ms_step in sorted(self.MILESTONES.keys()):
            if current_step >= ms_step:
                relevant_milestone = (ms_step, self.MILESTONES[ms_step])

        if relevant_milestone:
            ms_step, expected = relevant_milestone
            expected_wr = expected['eval/win_rate']
            tolerance = 0.07  # 7% tolerance (embeddings can have variable warmup)

            if current_wr < expected_wr - tolerance:
                issues.append({
                    'type': 'milestone_behind',
                    'severity': 'warning',
                    'message': (
                        f"At {current_step:,} steps: {current_wr:.1%} win rate "
                        f"< expected {expected_wr:.1%} (milestone: {ms_step:,})"
                    ),
                    'description': expected['description'],
                })
            elif current_wr >= expected_wr:
                # Positive milestone — log but don't flag as issue
                issues.append({
                    'type': 'milestone_reached',
                    'severity': 'info',
                    'message': (
                        f"✓ Milestone {ms_step:,}: {current_wr:.1%} ≥ {expected_wr:.1%} "
                        f"— {expected['description']}"
                    ),
                })

        return issues

    def detect_issues(self, metrics: dict) -> list:
        """Detect common training issues."""
        issues = []

        # Dead training (no policy updates happening)
        if 'train/clip_fraction' in metrics:
            recent = [v for _, v in metrics['train/clip_fraction'][-5:]]
            if recent and np.mean(recent) < 0.005:
                issues.append({
                    'type': 'dead_training',
                    'severity': 'critical',
                    'message': f'clip_fraction ≈ {np.mean(recent):.4f} — training stalled',
                    'suggestion': 'Increase learning rate or check if environment is stuck',
                })

        # Excessive clipping (updates too aggressive for clip_range=0.08)
        if 'train/clip_fraction' in metrics:
            recent = [v for _, v in metrics['train/clip_fraction'][-5:]]
            if recent and np.mean(recent) > 0.40:
                issues.append({
                    'type': 'excessive_clipping',
                    'severity': 'warning',
                    'message': f'clip_fraction ≈ {np.mean(recent):.2f} — updates too aggressive',
                    'suggestion': 'Reduce learning rate or n_epochs',
                })

        # Entropy monitoring (context-aware for 13-action space)
        if 'train/entropy_loss' in metrics:
            recent = [v for _, v in metrics['train/entropy_loss'][-5:]]
            if recent:
                avg_ent = np.mean(recent)
                # For 13 actions, entropy of -0.5 means ~2 actions (too deterministic early)
                # entropy of -2.2 means ~9 actions (too random late)
                current_step = metrics['train/entropy_loss'][-1][0]

                if current_step < 5_000_000 and avg_ent > -0.8:
                    issues.append({
                        'type': 'entropy_collapse_early',
                        'severity': 'warning',
                        'message': f'Entropy too low early in training ({avg_ent:.2f}, ≈{np.exp(-avg_ent):.0f} actions)',
                        'suggestion': 'Increase ent_coef (currently exploring too few actions)',
                    })
                elif current_step > 15_000_000 and avg_ent < -2.0:
                    issues.append({
                        'type': 'entropy_too_high_late',
                        'severity': 'info',
                        'message': f'Entropy still high late in training ({avg_ent:.2f}, ≈{np.exp(-avg_ent):.0f} actions)',
                        'suggestion': 'Policy may not be converging. Consider reducing ent_coef.',
                    })

        # Value function with gamma=0.9999 context
        if 'train/explained_variance' in metrics:
            recent = [v for _, v in metrics['train/explained_variance'][-5:]]
            if recent:
                avg_ev = np.mean(recent)
                current_step = metrics['train/explained_variance'][-1][0]

                if avg_ev < 0.0:
                    issues.append({
                        'type': 'negative_explained_variance',
                        'severity': 'warning',
                        'message': f'Explained variance negative ({avg_ev:.3f}) — value function hurting',
                        'suggestion': 'Value predictions worse than mean. Increase vf_coef or check rewards.',
                    })
                elif avg_ev < 0.15 and current_step > 3_000_000:
                    issues.append({
                        'type': 'poor_value_fn',
                        'severity': 'warning',
                        'message': f'Explained variance low ({avg_ev:.3f}) after {current_step/1e6:.0f}M steps',
                        'suggestion': (
                            'With gamma=0.9999, this is partially expected. '
                            'If win rate is still improving, this is OK.'
                        ),
                    })

        # KL divergence
        if 'train/approx_kl' in metrics:
            recent = [v for _, v in metrics['train/approx_kl'][-5:]]
            if recent:
                avg_kl = np.mean(recent)
                if avg_kl > 0.04:
                    issues.append({
                        'type': 'high_kl',
                        'severity': 'warning',
                        'message': f'KL divergence high ({avg_kl:.4f}) — updates too aggressive',
                        'suggestion': 'Reduce learning rate or n_epochs',
                    })
                elif avg_kl < 0.0005:
                    issues.append({
                        'type': 'low_kl',
                        'severity': 'warning',
                        'message': f'KL divergence very low ({avg_kl:.6f}) — barely learning',
                        'suggestion': 'Increase learning rate',
                    })

        # Check embedding growth (shared encoder should make them grow faster)
        for emb_name in ['pokemon', 'move', 'ability', 'item', 'type']:
            key = f'embeddings/{emb_name}_norm'
            if key in metrics and len(metrics[key]) >= 5:
                values = [v for _, v in metrics[key]]
                early = np.mean(values[:3])
                recent = np.mean(values[-3:])
                current_step = metrics[key][-1][0]

                # Explosion check
                if recent > 100:
                    issues.append({
                        'type': 'embedding_explosion',
                        'severity': 'critical',
                        'message': f'{emb_name} embedding norm exploding ({recent:.1f})',
                        'suggestion': 'Reduce learning rate or add weight decay',
                    })

                # Stagnation check (only warn after 2M steps)
                elif current_step > 2_000_000 and recent < early * 1.2:
                    issues.append({
                        'type': 'embedding_stagnant',
                        'severity': 'info',
                        'message': f'{emb_name} embeddings slow to grow ({early:.1f} → {recent:.1f})',
                        'suggestion': (
                            'Shared encoder gets 12x gradients, so some plateau is normal. '
                            'Check if win rate is still improving.'
                        ),
                    })

        if 'eval/win_rate' in metrics:
            values = [v for _, v in metrics['eval/win_rate']]
            steps = [s for s, _ in metrics['eval/win_rate']]

            if len(values) >= 5:
                recent_5 = values[-5:]
                current_step = steps[-1]

                # Plateau detection
                if max(recent_5) - min(recent_5) < 0.02:
                    plateau_val = np.mean(recent_5)
                    issues.append({
                        'type': 'plateau',
                        'severity': 'info' if plateau_val > 0.70 else 'warning',
                        'message': f'Win rate plateaued at {plateau_val:.1%} for last 5 evals',
                        'suggestion': (
                            'If >70%: consider adding self-play to push higher. '
                            'If <70%: check LR schedule, may need more training.'
                        ),
                    })

                # Regression detection
                if len(values) >= 3:
                    best_wr = max(values)
                    if values[-1] < best_wr - 0.10:
                        issues.append({
                            'type': 'regression',
                            'severity': 'warning',
                            'message': f'Win rate dropped: current {values[-1]:.1%} vs best {best_wr:.1%}',
                            'suggestion': 'May be temporary. If persistent, LR might be too high.',
                        })

        issues.extend(self.check_milestones(metrics))

        return issues

    def print_report(self, metrics: dict):
        """Print a formatted training report."""
        print("\n" + "=" * 70)
        print("TRAINING MONITOR — Pokemon RL v17")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log dir: {self.log_dir}")
        print("=" * 70)

        if not metrics:
            print("\n⚠ No metrics found. Is training running?")
            print(f"  Looking in: {self.log_dir}")
            return

        # Current step and progress
        all_steps = []
        for values in metrics.values():
            if values:
                all_steps.append(values[-1][0])

        if all_steps:
            current_step = max(all_steps)
            total_target = 30_000_000
            progress = current_step / total_target
            bar_len = 30
            filled = int(bar_len * progress)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"\nStep: {current_step:,} / {total_target:,}")
            print(f"Progress: [{bar}] {progress:.1%}")

        print("\n── Win Rate vs SimpleHeuristics ──")
        if 'eval/win_rate' in metrics and metrics['eval/win_rate']:
            best_wr = 0
            for step, wr in metrics['eval/win_rate']:
                best_wr = max(best_wr, wr)
                marker = " ★" if wr >= best_wr else ""

                # Show milestone status
                ms_status = ""
                for ms_step, ms_data in sorted(self.MILESTONES.items()):
                    if step >= ms_step * 0.9 and step <= ms_step * 1.1:
                        if wr >= ms_data['eval/win_rate']:
                            ms_status = f"  (✓ {ms_step/1e6:.0f}M target: {ms_data['eval/win_rate']:.0%})"
                        else:
                            ms_status = f"  ({ms_step/1e6:.0f}M target: {ms_data['eval/win_rate']:.0%})"
                        break

                print(f"  {step:>10,}: {wr:>5.1%}{marker}{ms_status}")

            print(f"\n  Best: {best_wr:.1%}  |  v13 baseline: 72%  |  Target: 78%+")
        else:
            print("  No eval data yet")

        print("\n── PPO Metrics ──")
        ppo_metrics = [
            'train/clip_fraction',
            'train/approx_kl',
            'train/entropy_loss',
            'train/explained_variance',
            'train/value_loss',
            'train/learning_rate',
        ]
        for name in ppo_metrics:
            if name in metrics:
                analysis = self.analyze_metric(name, metrics[name])
                short = name.split('/')[-1]
                print(f"  {short:25s}: {analysis['message']}")

        print("\n── Embedding Norms ──")
        emb_metrics = [
            'embeddings/pokemon_norm',
            'embeddings/move_norm',
            'embeddings/ability_norm',
            'embeddings/item_norm',
            'embeddings/type_norm',
        ]
        any_emb = False
        for name in emb_metrics:
            if name in metrics:
                any_emb = True
                analysis = self.analyze_metric(name, metrics[name])
                short = name.split('/')[-1]
                print(f"  {short:25s}: {analysis['message']}")
        if not any_emb:
            print("  (no embedding data yet)")

        if 'train/win_rate' in metrics:
            analysis = self.analyze_metric('train/win_rate', metrics['train/win_rate'])
            print(f"\n── Training Win Rate ──")
            print(f"  {'win_rate':25s}: {analysis['message']}")

        issues = self.detect_issues(metrics)

        critical = [i for i in issues if i['severity'] == 'critical']
        warnings = [i for i in issues if i['severity'] == 'warning']
        infos = [i for i in issues if i['severity'] == 'info']

        if critical or warnings:
            print(f"\n── Issues ({len(critical)} critical, {len(warnings)} warnings) ──")
            for issue in critical:
                print(f"  ✗ [{issue['type']}] {issue['message']}")
                if 'suggestion' in issue:
                    print(f"    → {issue['suggestion']}")
            for issue in warnings:
                print(f"  ⚠ [{issue['type']}] {issue['message']}")
                if 'suggestion' in issue:
                    print(f"    → {issue['suggestion']}")
        else:
            print(f"\n── No Issues ──")
            print("  ✓ Training appears healthy")

        if infos:
            print(f"\n── Notes ──")
            for issue in infos:
                print(f"  ℹ {issue['message']}")

        print(f"\n── v17 Quick Reference ──")
        print(f"  clip_fraction 15-30%: normal for clip_range=0.08")
        print(f"  entropy_loss -1.5: ≈5 actions explored (good)")
        print(f"  explained_var 0.15+: acceptable with gamma=0.9999")
        print(f"  embedding norms 5-40: healthy range for shared encoder")

        print("\n" + "=" * 70)

    def compare_runs(self, run_names: list):
        """Compare metrics across multiple runs."""
        print("\n" + "=" * 70)
        print("RUN COMPARISON")
        print("=" * 70)

        all_metrics = {}
        for name in run_names:
            metrics = self.load_tensorboard_logs(run_name=name)
            if metrics:
                all_metrics[name] = metrics
            else:
                print(f"  ⚠ No data found for run: {name}")

        if len(all_metrics) < 2:
            print("  Need at least 2 runs to compare.")
            return

        # Compare win rates
        print("\n── Win Rate Comparison ──")
        for name, metrics in all_metrics.items():
            if 'eval/win_rate' in metrics:
                values = metrics['eval/win_rate']
                best = max(v for _, v in values)
                latest = values[-1][1]
                steps = values[-1][0]
                print(f"  {name:15s}: latest={latest:.1%}  best={best:.1%}  @ {steps:,} steps")

        # Compare key metrics
        print("\n── Latest Metrics ──")
        compare_keys = ['train/entropy_loss', 'train/explained_variance', 'train/approx_kl']
        for key in compare_keys:
            short = key.split('/')[-1]
            row = f"  {short:25s}:"
            for name, metrics in all_metrics.items():
                if key in metrics and metrics[key]:
                    val = metrics[key][-1][1]
                    row += f"  {name}={val:.4f}"
            print(row)

        print("\n" + "=" * 70)

    def watch(self, interval: int = 60):
        """Continuously watch training progress."""
        print(f"Watching {self.log_dir} (refresh: {interval}s)")
        print("Press Ctrl+C to stop\n")
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                metrics = self.load_tensorboard_logs()
                self.print_report(metrics)
                print(f"\nRefreshing in {interval}s...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")

    def single_report(self):
        metrics = self.load_tensorboard_logs()
        self.print_report(metrics)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Monitor Pokemon RL v17 training')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='TensorBoard log directory')
    parser.add_argument('--watch', action='store_true',
                        help='Continuously monitor')
    parser.add_argument('--interval', type=int, default=60,
                        help='Refresh interval for --watch')
    parser.add_argument('--compare', nargs='+',
                        help='Compare runs (e.g., --compare v13 v17)')

    args = parser.parse_args()

    monitor = TrainingMonitor(log_dir=args.log_dir)

    if args.compare:
        monitor.compare_runs(args.compare)
    elif args.watch:
        monitor.watch(interval=args.interval)
    else:
        monitor.single_report()


if __name__ == "__main__":
    main()