"""Thin adapter for the shared, frozen A/B curriculum runner."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'Capri/src'))
from capri_controller.training.ab_curriculum import main
if __name__ == '__main__':
    sys.argv[1:1]=['train','--model','2']
    main()
