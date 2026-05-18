"""
Experiment Tracker
Records each training run's parameters, metrics, and metadata for reproducibility.
Uses JSONL format for append-only logging.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Track ML experiment runs with parameters, metrics, and metadata."""

    def __init__(self, experiment_dir: str = "experiments"):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.experiment_dir / "runs.jsonl"

    def log_run(
        self,
        experiment_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        feature_names: List[str],
        feature_groups: List[str] = None,
        notes: str = "",
        tags: List[str] = None,
    ) -> str:
        """Log a single experiment run.

        Args:
            experiment_name: Human-readable experiment name.
            params: Training parameters (model type, hyperparams, etc.).
            metrics: Evaluation metrics (AUC, Brier, etc.).
            feature_names: List of feature names used.
            feature_groups: Feature group names if using ablation.
            notes: Free-text notes.
            tags: Tags for filtering (e.g., ['ablation', 'model_c']).

        Returns:
            Experiment run ID.
        """
        run_id = str(uuid.uuid4())[:8]

        record = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'experiment_name': experiment_name,
            'params': params,
            'metrics': metrics,
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'feature_groups': feature_groups,
            'notes': notes,
            'tags': tags or [],
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')

        logger.info(f"Logged experiment run {run_id}: {experiment_name}")
        return run_id

    def get_runs(
        self,
        experiment_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Query experiment runs with optional filters.

        Args:
            experiment_name: Filter by experiment name.
            tags: Filter by tags (any match).
            limit: Max results.

        Returns:
            List of matching run records.
        """
        if not self.log_file.exists():
            return []

        runs = []
        with open(self.log_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if experiment_name and record.get('experiment_name') != experiment_name:
                    continue
                if tags and not set(tags).intersection(set(record.get('tags', []))):
                    continue

                runs.append(record)

        return runs[-limit:]

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare metrics across specific runs.

        Args:
            run_ids: List of run IDs to compare.

        Returns:
            Comparison dict with metrics side by side.
        """
        all_runs = self.get_runs(limit=10000)
        selected = [r for r in all_runs if r.get('run_id') in run_ids]

        if not selected:
            return {'error': 'No matching runs found'}

        comparison = {
            'runs': selected,
            'metric_comparison': {},
        }

        # Collect all metric keys
        all_metric_keys = set()
        for run in selected:
            all_metric_keys.update(run.get('metrics', {}).keys())

        for key in sorted(all_metric_keys):
            values = {}
            for run in selected:
                val = run.get('metrics', {}).get(key)
                if val is not None:
                    values[run['run_id']] = val
            comparison['metric_comparison'][key] = values

        return comparison

    def get_best_run(
        self,
        experiment_name: str,
        metric: str = 'auc',
        higher_is_better: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Find the best run for an experiment by a specific metric."""
        runs = self.get_runs(experiment_name=experiment_name)
        if not runs:
            return None

        def get_metric_value(run):
            m = run.get('metrics', {})
            # Handle nested structure (e.g., {'lightgbm': {'auc': 0.75}})
            if isinstance(m, dict):
                for v in m.values():
                    if isinstance(v, dict) and metric in v:
                        return v[metric]
                return m.get(metric)
            return None

        valid_runs = [(r, get_metric_value(r)) for r in runs]
        valid_runs = [(r, v) for r, v in valid_runs if v is not None]

        if not valid_runs:
            return None

        return max(valid_runs, key=lambda x: x[1] if higher_is_better else -x[1])[0]
