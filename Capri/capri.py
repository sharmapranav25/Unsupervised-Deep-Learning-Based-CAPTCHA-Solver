"""Single entry point for the maintained CAPRI controller workflow."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from capri_controller.cli import main
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='train-cryptbb-model1':
        del sys.argv[1]
        from capri_controller.training.cryptbb_model1 import main as train_cryptbb
        train_cryptbb()
    elif len(sys.argv)>1 and sys.argv[1]=='finish-cryptbb-preparation':
        del sys.argv[1]
        from capri_controller.cryptbb_preflight_repair import main as finish_cryptbb
        finish_cryptbb()
    elif len(sys.argv)>1 and sys.argv[1]=='prepare-cryptbb-model1':
        del sys.argv[1]
        from capri_controller.cryptbb_preparation import main as cryptbb
        cryptbb()
    elif len(sys.argv)>1 and sys.argv[1]=='controller4-validation':
        del sys.argv[1]
        from capri_controller.synthtiger_validation import main as controller4
        controller4()
    elif len(sys.argv)>1 and sys.argv[1]=='darkbay-ab-curriculum':
        del sys.argv[1]
        from capri_controller.training.ab_curriculum import main as ab_curriculum
        ab_curriculum()
    elif len(sys.argv)>1 and sys.argv[1]=='test-lightweight-count':
        del sys.argv[1]
        from capri_controller.lightweight_count_evaluation import main as evaluate_count
        evaluate_count()
    elif len(sys.argv)>1 and sys.argv[1]=='compare-capri-count':
        del sys.argv[1]
        from capri_controller.count_baseline_comparison import main as compare_count
        compare_count()
    elif len(sys.argv)>1 and sys.argv[1]=='compare-capri-controllers':
        del sys.argv[1]
        from capri_controller.controller_comparison import main as compare_controllers
        compare_controllers()
    elif len(sys.argv)>1 and sys.argv[1]=='train-capri2-parseq':
        del sys.argv[1]
        from capri_controller.training.capri2_parseq import main as capri2_parseq
        capri2_parseq()
    else:main()

