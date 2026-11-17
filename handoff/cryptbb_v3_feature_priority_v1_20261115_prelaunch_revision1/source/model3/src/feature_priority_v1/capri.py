"""Versioned Model3 launcher; preserves the historical CAPRI entry point."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Capri/src'))
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='inventory':
        from capri_controller.inventory import refresh
        refresh()
    else:
        from capri_controller.training.cryptbb_feature_v1 import main
        main()
