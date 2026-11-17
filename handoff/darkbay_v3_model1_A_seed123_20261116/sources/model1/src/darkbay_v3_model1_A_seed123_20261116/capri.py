from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Capri/src'))
if __name__=='__main__':
    from capri_controller.training.darkbay_model1_seed123 import main
    main()
