"""Complete the adopted validation-only selection sweep, preserving its earlier snapshot."""
from pathlib import Path
import evaluate_controller4_validation as sweep
sweep.OUT=sweep.core.STUDY/'controller4_validation_sweep_final'
sweep.FIG=sweep.ROOT/'thesis/figures/darkbay/controller4_validation_final'
if __name__=='__main__':
    for tag in ('A','B'):
        done=sweep.core.read(sweep.ROOT/f'model3/runs/darkbay/{tag}/completion.json')
        assert done['epoch']==20 and done['global_step']==15640
    sweep.OUT.mkdir(parents=True,exist_ok=True)
    sweep.core.write(sweep.OUT/'completion_addendum.json',dict(wrapper_sha256=sweep.core.sha(Path(__file__)),base_evaluator_sha256=sweep.core.sha(Path(sweep.__file__)),prior_snapshot='handoff/darkbay_curriculum_20261113/controller4_validation_sweep',policy='handoff/darkbay_curriculum_20261113/validation_selection_policy.json',method='Identical frozen evaluator and dataset; reuse existing112 evaluations and add M3 A/B18-20; preserve prior reports.'))
    sweep.main()
