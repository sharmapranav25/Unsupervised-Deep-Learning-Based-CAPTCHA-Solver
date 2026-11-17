"""Single entry point for the maintained CAPRI controller workflow."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from capri_controller.cli import main
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='test-lightweight-count':
        del sys.argv[1]
        from capri_controller.lightweight_count_evaluation import main as evaluate_count
        evaluate_count()
    elif len(sys.argv)>1 and sys.argv[1]=='compare-capri-count':
        del sys.argv[1]
        from capri_controller.count_baseline_comparison import main as compare_count
        compare_count()
    else:main()
