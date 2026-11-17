"""Explicit commands; inspection never starts training or overwrites a run."""
import argparse,json,os,sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description='CAPRI pixel-only controller workflow')
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare-lightweight','check-lightweight','train-lightweight'):
        cmd=commands.add_parser(name);cmd.add_argument('--dataset',required=True);cmd.add_argument('--out',required=True)
        if name=='train-lightweight':cmd.add_argument('--readiness',required=True);cmd.add_argument('--resume',action='store_true')
    for name in ('check-joint20','train-joint20'):
        cmd=commands.add_parser(name)
        cmd.add_argument('--dataset',required=True);cmd.add_argument('--out',required=True)
        if name=='check-joint20':cmd.add_argument('--batch-size',type=int,default=8)
        else:cmd.add_argument('--readiness',required=True);cmd.add_argument('--resume',action='store_true')
    for name in ('check-oldstyle-100k','train-oldstyle-100k'):
        cmd=commands.add_parser(name)
        cmd.add_argument('--dataset',required=True);cmd.add_argument('--out',required=True)
        if name=='train-oldstyle-100k':cmd.add_argument('--readiness',required=True);cmd.add_argument('--resume',action='store_true')
    for name in ('check-architecture-training','check-architecture-evaluation','train-architecture-100k'):
        cmd=commands.add_parser(name)
        cmd.add_argument('--dataset',required=True);cmd.add_argument('--out',required=True)
        if name!='train-architecture-100k':cmd.add_argument('--checkpoint',required=True)
        else:cmd.add_argument('--readiness',required=True);cmd.add_argument('--resume',action='store_true')
    large=commands.add_parser('prepare-architecture-100k',help='Prepare exactly100k TRAIN images plus separate DEV; no training')
    large.add_argument('--source',default='runs/capri_decoder_500k_20261007')
    large.add_argument('--base',default='runs/capri_shared_count_dataset_20261027')
    large.add_argument('--out',required=True)
    large.add_argument('--workers',type=int,default=4)
    prep=commands.add_parser('prepare-architecture-data',help='Prepare portable shared-CAPRI and independent-count data; no training')
    prep.add_argument('--source',default='runs/capri_decoder_500k_20261007')
    prep.add_argument('--out',required=True)
    final=commands.add_parser('finalize-architecture-data',help='Recover verified data stopped on reference pixel ambiguity')
    final.add_argument('--source',default='runs/capri_decoder_500k_20261007')
    final.add_argument('--dataset',required=True)
    check=commands.add_parser('verify-architecture-data',help='Verify prepared architecture data; no training')
    check.add_argument('--dataset',required=True)
    commands.add_parser('status',help='Show active run and quality results')
    commands.add_parser('models',help='List selected checkpoints and their roles')
    commands.add_parser('verify',help='Verify all active checkpoint hashes and external dependencies')
    commands.add_parser('inventory',help='Refresh the workspace/model catalog without moving files')
    commands.add_parser('test-decoder-real',help='Freeze final CAPRI pilot predictions on historical real DarkBay, then score count only')
    commands.add_parser('test-decoder-schemes',help='Evaluate 100 pixels-only samples each from CryptBB, City and SimpleCaptcha; render fixed examples')
    commands.add_parser('render-decoder-schemes',help='Export and verify previews from frozen scheme predictions; no model inference')
    commands.add_parser('score-decoder-schemes',help='Score existing frozen predictions from label hashes and published count metadata')
    decoder=commands.add_parser('decoder',help='CAPRI Decoder data/model pipeline; separate from existing controllers')
    decoder.add_argument('stage',choices=['timed-pilot','timed-pilot-worker','profile-stopped','audit-dev-gallery','dev-gallery-reader-tests','dev-gallery-tests','generate-dev-gallery','joint-phase-tests','reference-sampling-tests','font-training-tests','observe-training','migrate-storage','amp-recovery-tests','joint-evaluation-tests','font-extraction-tests','phase-engine-tests','specialist-selection-tests','reference-data-tests','specialist-contract-tests','acquire-dinov2','train-baseline','training-tests','status','supervise','audit-corpus','affinity-tests','diagnose-render-failure','recovery-tests','model-tests','model-resume-tests','appearance-tests','retrieval-tests','prepare-font-references','reference-pilot','audit-font-references','reference-diagnose','reference-recovery-tests','reference-storage-tests','generate-font-references','acquire-fonts','archive-fonts','historical-fonts','distro-fonts','index-fonts','renderer-tests','checkpoint-tests','checkpoint-cuda-tests','objective-tests','pilot','storage-tests','data-tests','prepare-corpus','generate-corpus'])
    decoder.add_argument('--run',default='runs/capri_decoder_500k_20261007')
    decoder.add_argument('--workers',type=int,default=8)
    decoder.add_argument('--count',type=int,default=256)
    commands.add_parser('prepare-parseq',help='Prepare all100k DarkBay images for the fixed PARSeq protocol')
    commands.add_parser('test-parseq',help='Freeze and score all three final readers on real DarkBay; refuses to overwrite an existing evaluation')
    commands.add_parser('darkbay-ds2',help='Generate DS2, score curriculum, train three20-epoch readers, then evaluate')
    ds3=commands.add_parser('darkbay-ds3',help='Adaptive curriculum on the unchanged DS2 corpus and split')
    ds3.add_argument('--stage',choices=['prepare','smoke','train','run','evaluate'],default='run')
    ds3.add_argument('--seed',type=int,choices=[48,123,456],default=48)
    ds3.add_argument('--resume',action='store_true')
    ds3.add_argument('--stop-after',type=int)
    reader=commands.add_parser('train-parseq',help='Train the fixed20-epoch PARSeq experiment; requires explicit initialization')
    initialization=reader.add_mutually_exclusive_group(required=True)
    initialization.add_argument('--checkpoint');initialization.add_argument('--from-scratch',action='store_true')
    reader.add_argument('--resume',action='store_true')
    controllers=commands.add_parser('controllers',help='Create a fresh 100-input controller run')
    controllers.add_argument('--out',required=True);controllers.add_argument('--targets',help='JSON manifest with a fit list of source/pixel_sha256 objects')
    controllers.add_argument('--preview',action='store_true',help='Refine first6 inputs against all100 references; enable documented equal-observed-length fallback; no corpus export')
    commands.add_parser('preview-smaller-hybrid',help='First6 examples for each saved100-image smaller-hybrid set using all three controllers')
    commands.add_parser('review-smaller-hybrid',help='Export controller comparisons including explicit failures and replay saved successful partial entries')
    for name in ['generate','audit','report']:
        command=commands.add_parser(name,help={'generate':'Export a corpus','audit':'Decode and audit every corpus image','report':'Build a lightweight run overview'}[name])
        command.add_argument('--run',required=True)
        if name=='generate':command.add_argument('--workers',type=int,default=8);command.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare-lightweight':
        from .lightweight.data import prepare
        prepare(args.dataset,args.out);return
    if args.command in ('check-lightweight','train-lightweight'):
        from .lightweight.training import readiness,run
        if args.command=='check-lightweight':readiness(args.dataset,args.out)
        else:run(args.dataset,args.out,args.readiness,args.resume)
        return
    if args.command=='check-joint20':
        from .joint_training_20 import readiness
        readiness(args.dataset,args.out,args.batch_size);return
    if args.command=='train-joint20':
        from .joint_training_20 import run
        run(args.dataset,args.out,args.readiness,args.resume);return
    if args.command=='check-oldstyle-100k':
        from .oldstyle_training import readiness
        readiness(args.dataset,args.out);return
    if args.command=='train-oldstyle-100k':
        from .oldstyle_training import run
        run(args.dataset,args.out,args.readiness,args.resume);return
    if args.command=='check-architecture-evaluation':
        from .architecture_training_checks import run
        run(args.dataset,args.out,Path(args.checkpoint).resolve());return
    if args.command=='check-architecture-training':
        from .architecture_training import readiness
        readiness(args.dataset,args.out,args.checkpoint);return
    if args.command=='train-architecture-100k':
        from .architecture_training import run
        run(args.dataset,args.out,args.readiness,args.resume);return
    if args.command=='prepare-architecture-100k':
        from .architecture_data_100k import prepare
        print(json.dumps(prepare(args.source,args.base,args.out,args.workers),indent=2));return
    if args.command=='prepare-architecture-data':
        from .architecture_data import prepare
        print(json.dumps(prepare(args.source,args.out),indent=2));return
    if args.command=='finalize-architecture-data':
        from .architecture_data import finalize_existing
        print(json.dumps(finalize_existing(args.source,args.dataset),indent=2));return
    if args.command=='verify-architecture-data':
        from .architecture_data import verify
        print(json.dumps(verify(args.dataset),indent=2));return

    if args.command=='controllers':
        output=Path(args.out).resolve()
        if output.exists():parser.error('--out must be a new directory; completed runs are preserved')
        os.environ['CAPRI_RUN_DIR']=str(output)
        if args.targets:os.environ['CAPRI_TARGET_MANIFEST']=str(Path(args.targets).resolve())
    elif hasattr(args,'run') and args.command!='decoder':
        run=Path(args.run).resolve()
        if not (run/'entries.json').is_file():parser.error('--run must contain entries.json')
        if args.command=='generate' and (run/'protocol.json').exists() and json.loads((run/'protocol.json').read_text()).get('preview_only'):
            parser.error('This run is a bounded controller preview; corpus generation is not enabled.')
        os.environ['CAPRI_RUN_DIR']=str(run)
    from . import runtime as U
    if args.command=='review-smaller-hybrid':
        from .smaller_hybrid_preview_review import main as review_hybrid
        review_hybrid();return
    if args.command=='preview-smaller-hybrid':
        from .smaller_hybrid_preview import main as preview_hybrid
        preview_hybrid();return
    if args.command=='score-decoder-schemes':
        from .decoder_scheme_scoring import main as score_schemes
        score_schemes();return
    if args.command=='render-decoder-schemes':
        from .decoder_scheme_previews import main as render_schemes
        render_schemes();return
    if args.command=='test-decoder-schemes':
        from .decoder_scheme_evaluation import main as evaluate_schemes
        evaluate_schemes();return
    if args.command=='test-decoder-real':
        from .decoder_real_evaluation import main as evaluate_real
        evaluate_real();return
    if args.command=='decoder':
        from .decoder.pipeline import main as decoder_main
        decoder_main(args);return
    if args.command=='darkbay-ds3':
        from .training.ds3 import main as ds3_main
        sys.argv=['darkbay-ds3',args.stage,'--seed',str(args.seed)]+(['--resume'] if args.resume else [])+(['--stop-after',str(args.stop_after)] if args.stop_after else [])
        ds3_main();return
    if args.command=='darkbay-ds2':
        from .training.ds2 import main as ds2
        sys.argv=['darkbay-ds2','run'];ds2();return
    if args.command=='status':
        result={'run':str(U.RUN_ROOT),'controllers':['custom','dino','mixed'],'no_vlms':True}
        for name in ['status','length_decision','corpus_audit','diversity']:
            path=U.RUN_ROOT/(name+'.json')
            if path.exists():result[name]=U.read(path)
        print(json.dumps(result,indent=2));return
    if args.command=='models':print(json.dumps(U.registry(),indent=2));return
    if args.command=='verify':
        result={'models':{},'dependencies':{str(p):p.is_dir() for p in [U.PROJECT,U.RENDERER]}}
        for name,entry in U.registry()['models'].items():
            path=U.model_path(name);result['models'][name]=path.is_file() and U.sha(path)==entry['sha256']
        result['ok']=all(result['models'].values()) and all(result['dependencies'].values())
        print(json.dumps(result,indent=2));raise SystemExit(0 if result['ok'] else 1)
    if args.command=='inventory':
        from .inventory import refresh
        refresh();return
    if args.command=='prepare-parseq':
        from .training.prepare_parseq import prepare
        prepare();return
    if args.command=='test-parseq':
        from .training.evaluate_parseq import main as evaluate
        evaluate();return
    if args.command=='train-parseq':
        from .training.train_parseq import main as train
        sys.argv=['train-parseq']+(['--checkpoint',args.checkpoint] if args.checkpoint else ['--from-scratch'])+(['--resume'] if args.resume else [])
        train();return
    if args.command=='report':
        from .reporting import build
        print(build(U.RUN_ROOT));return
    # Import expensive neural/renderer code only for operational commands.
    if args.command=='controllers':
        from .pipeline import main as run
        run(preview=args.preview);return
    from . import corpus
    if args.command=='audit':corpus.audit();return
    if args.workers<1:parser.error('--workers must be positive')
    sys.argv=['capri generate','--workers',str(args.workers)]+(['--pilot'] if args.pilot else [])
    corpus.main()
