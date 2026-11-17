"""A/B replacement for historical evaluate_ds2_checkpoint.py path assumptions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from capri_controller.training.ab_status_evaluation import main
if __name__=='__main__':main()
