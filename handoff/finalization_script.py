"""Build current documentation and portable manifests after the verified copy."""
import copy
import importlib.util
import json
import shutil
from pathlib import Path
import yaml

spec=importlib.util.spec_from_file_location('migration',Path(__file__).with_name('migrate_thesis_final_20261112.py'))
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
R=M.DST

def doc(path,text):
    p=R/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text.strip()+'\n',encoding='utf8')
def dump(path,value):M.write(R/path,value)

def main():
    assert (R/'handoff/migration_copy_complete.json').exists()
    # Resolve sourcefree_synth's import and font/license references, beyond its core package.
    M.plan.clear()
    M.tree(M.SF/'feature_bank','Capri/dependencies/sourcefree_synth/feature_bank')
    for catalog in (M.SF/'resources').rglob('fonts.json'):
        entries=M.read(catalog)
        if not isinstance(entries,list):continue
        for entry in entries:
            if not isinstance(entry,dict):continue
            for key in ('path','license_path'):
                val=entry.get(key)
                if not val:continue
                p=M.SF/val
                if p.is_file() and p.is_relative_to(M.SF):
                    relative='Capri/dependencies/sourcefree_synth/'+p.relative_to(M.SF).as_posix()
                    if not (R/relative).exists():M.add(p,relative)
    manifest=R/'handoff/COPY_MANIFEST.jsonl'
    known={row['destination'] for row in map(json.loads,manifest.read_text(encoding='utf8').splitlines())}
    with manifest.open('a',encoding='utf8') as log:
        for relative,row in M.plan.items():
            if relative in known:continue
            s=Path(row['source']);d=R/relative;digest=M.sha(s)
            d.parent.mkdir(parents=True,exist_ok=True)
            if not d.exists():shutil.copy2(s,d)
            assert M.sha(d)==digest
            log.write(json.dumps(dict(**row,sha256=digest,verified_at=M.now()))+'\n')
    records=list(map(json.loads,manifest.read_text(encoding='utf8').splitlines()))
    mapping={r['source']:r['destination'] for r in records}
    # Resolve final aliases to one physical checkpoint; archive serialization hashes may differ.
    import torch
    torch.set_num_threads(4)
    equivalence={}
    for tag,run in [('A',M.A),('B',M.B)]:
        evaluated=torch.load(run/'seed_48/latest.pth',map_location='cpu',weights_only=False)
        archived=torch.load(run/'seed_48/epoch_20.pth',map_location='cpu',weights_only=False)
        assert evaluated['epoch']==archived['epoch']==20
        assert evaluated['global_step']==archived['global_step']==15640
        assert evaluated['config']==archived['config']
        assert evaluated['state_dict'].keys()==archived['state_dict'].keys()
        assert all(torch.equal(v,archived['state_dict'][k]) for k,v in evaluated['state_dict'].items())
        equivalence[tag]=dict(weights_exactly_equal=True,config_equal=True,epoch=20,global_step=15640,
            evaluated_latest_sha256=M.sha(run/'seed_48/latest.pth'),archived_epoch20_sha256=M.sha(run/'seed_48/epoch_20.pth'))
        del evaluated,archived
        mapping[str(run/'seed_48/latest.pth')]=f'model1/models/parseq/darkbay/{tag}/epoch_20.pth'
    dump('handoff/final_checkpoint_equivalence.json',equivalence)
    mapping[str(M.GEN/'epochs/epoch_0020/checkpoint.pt')]='Capri/models/capri/checkpoint.pt'
    dump('handoff/SOURCE_MAP.json',dict(root=str(R),copied_files_manifest='handoff/COPY_MANIFEST.jsonl',files=mapping,
        original_files_preserved=True,historical_path_policy='Original protocols and embedded checkpoint configurations are immutable. Use these mappings and current manifests; C: strings in historical evidence are provenance, not new write destinations.',
        historical_c_prefix_alias={'C:/Users/pranav/Documents/ChatGPT/thesis/':'E:/pranav/CAPRI/'},
        external_runtime='Existing E: Python environment is reused; not bundled or portable to another host without rebuilding dependencies.'))
    cfg=M.read(M.SRC/'configs/local.json')
    cfg.update(project=str(R/'Capri/dependencies/sourcefree_synth'),renderer=str(R/'Capri/renderer'),
        default_run='runs/new_run',target_manifest='configs/targets/darkbay_final.json',
        decoder_dinov2_source=str(R/'Capri/src/capri_controller/vendor/dinov2-85a24602099d397264d5b30461ad7f3bfd726ca1'),
        decoder_dinov2_checkpoint=str(R/'Capri/models/pretrained/dinov2_vits14_reg4_pretrain.pth'))
    for key in ('decoder_storage','decoder_run','decoder_download_python','decoder_dinov3_source','decoder_scheme_inputs'):cfg.pop(key,None)
    dump('Capri/configs/local.json',cfg)
    registry=M.read(M.SRC/'models/registry.json')
    registry.pop('reader_experiments',None)
    for name,entry in registry['models'].items():
        entry['path']=entry['path'].replace('models/active/','models/controller_dependencies/')
        if name=='hybrid':entry['role']='Legacy calibration dependency only; NOT the selected thesis generator.'
    registry['selected_generator']=dict(path='models/capri/checkpoint.pt',sha256=M.read(M.GEN/'completion.json')['sha256'],epoch=20,weights='EMA',internal_count=True,separate_counter_used=False)
    registry['entrypoint_limit']='Historical generator orchestration must be adapted before new runs; selected_generator is authoritative, generic legacy hybrid routing is not.'
    dump('Capri/models/registry.json',registry)
    refs=M.read(M.A/'targets.json')['fit']
    for row in refs:row['source']=str(R/'Capri/controllers/darkbay/reference_images'/Path(row['source']).name)
    dump('Capri/configs/targets/darkbay_final.json',{'fit':refs})
    original=M.read(M.SRC/'runs/parseq_darkbay_100k_20261005/evaluation_darkbay/image_manifest.json')
    for row in original:row['image_path']='evaluation_data/darkbay/images/'+Path(row['image_path']).name
    dump('evaluation_data/darkbay/split_manifests/image_manifest.json',original)
    datasets={}
    for tag,run in [('A',M.A),('B',M.B)]:
        receipt=M.read(run/'data_receipt.json')
        datasets[tag]=dict(images=100000,lmdb=f'datasets/darkbay/{tag}/train_lmdb',corpus=f'datasets/darkbay/{tag}/corpus',charset=f'datasets/darkbay/{tag}/charset.txt',lmdb_sha256=receipt['lmdb_sha256'],generator_sha256=registry['selected_generator']['sha256'],controllers=receipt['controllers'])
        c=yaml.safe_load((run/'seed_48.yml').read_text())
        def relocate(v):
            if isinstance(v,dict):return {k:relocate(x) for k,x in v.items()}
            if isinstance(v,list):return [relocate(x) for x in v]
            if isinstance(v,str):
                for old,new in ((str(run/'train_lmdb'),str(R/f'datasets/darkbay/{tag}/train_lmdb')),(str(run/'charset.txt'),str(R/f'datasets/darkbay/{tag}/charset.txt')),(str(run/'seed_48'),str(R/f'model1/runs/reproduction_{tag}'))):v=v.replace(old,new)
            return v
        c=relocate(c)
        (R/f'model1/configs/darkbay/{tag}/relocated.yml').write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf8')
        dump(f'model1/models/parseq/darkbay/{tag}/manifest.json',dict(path=f'model1/models/parseq/darkbay/{tag}/epoch_20.pth',sha256=equivalence[tag]['archived_epoch20_sha256'],evaluated_latest_sha256=equivalence[tag]['evaluated_latest_sha256'],equivalence_evidence='handoff/final_checkpoint_equivalence.json',epoch=20,dataset=tag,selection='Fixed final epoch20',original_checkpoint=str(run/'seed_48/epoch_20.pth')))
    dump('datasets/darkbay/manifests/datasets.json',datasets)
    dump('datasets/darkbay/manifests/audit.json',M.read(M.STUDY/'datasets_ready.json'))
    for model in ('model1','model2','model3'):
        dump(model+'/data_manifest.json',dict(dataset_manifest='datasets/darkbay/manifests/datasets.json',directions={'A':dict(dataset='A',teacher=None if model=='model1' else 'model1/models/parseq/darkbay/B/epoch_20.pth'),'B':dict(dataset='B',teacher=None if model=='model1' else 'model1/models/parseq/darkbay/A/epoch_20.pth')},status='complete' if model=='model1' else 'not_configured_not_started'))
    dump('runtime/local.json',dict(python=cfg['python'],workspace=str(R),cache='runtime/cache',temporary='runtime/temporary',openocr='model1/src/OpenOCR',parseq_packages='model1/src/parseq_packages',capri_source='Capri/src'))
    doc('README.md',r'''# Thesis-Pranav-Final

This is the shared thesis workspace on E:. Start with handoff/CURRENT_STATE.md. All original experiments are preserved; the files here were copied and SHA-256 verified. Use workspace-relative paths in current manifests. Immutable historical files retain their original paths and statements; they are not current instructions.

## Current study

One selected CAPRI generator: original lightweight core, fine-tuned DINO font branch and separate count transformer. Generation uses its internal count head; the separate transformer is retained as training provenance. A and B each contain 100,000 synthetic DarkBay training images. Their pixels are disjoint; their controller recipes and mixture are shared. The archive and LMDB are two representations of each corpus, shared across all reader approaches.

Model 1-A and 1-B are complete at epoch20. Same900 historical real DarkBay images: A615/900 (68.33%), B573/900 (63.67%). Models 2 and 3 have historical source code and evidence, but their new A/B rankings and training configurations are not prepared. No training is running or automatically authorized by this migration.

## Layout

- Capri/: generator, shared maintained package, controller dependencies, renderer, font assets, training data, code and provenance.
- datasets/darkbay/A and B: the only working copies of these corpora in this workspace.
- model1/: completed standard readers, OpenOCR, configs, checkpoints, evaluations.
- model2/: static curriculum; rankings are pending.
- model3/: adaptive curriculum; updates and training are pending.
- evaluation_data/: real image pixels and separate scoring labels; never training inputs.
- thesis/: empty writing, figures, tables and bibliography destinations. No manuscript was invented.
- handoff/: current state, source mappings, copy receipts, evidence and original documentation.
- runtime/: host Python location and E-drive caches. Python itself remains in the existing E: environment.

From PowerShell, source .\environment.ps1, then use `& $thesisPython -B workspace.py status`, `verify`, or `inventory`. `verify --full` rehashes all copied files; modified working sources will be reported against the original migration snapshot. These commands do not train. Use model1/src/verify_migrated_readers.py for the migration inference check.

Historical scripts have fixed run names and old orchestration assumptions. Do not launch them unchanged: the new A/B curriculum adapters are still pending. Canonical common implementation is Capri/src/capri_controller; model folders refer to it rather than maintaining divergent copies. The copied Capri/capri.py is preserved historical source, not an approved new-study launcher. Read Capri/README.md before using it.

This workspace is usable by another assistant on this machine. It is not yet a standalone Python environment for another computer: see runtime/local.json and handoff/DEPENDENCIES.md.
''')
    doc('AGENTS.md',r'''# Shared thesis workspace rules

Read README.md and handoff/CURRENT_STATE.md first. All work, outputs, caches and temporary project files stay on E:. Do not use private assistant workspaces for thesis artifacts. Record changes in handoff/CHANGELOG.md and maintain source/evidence mappings.

Maintain shared Python implementation in Capri/src/capri_controller, selected generator models in Capri/models/registry.json and machine paths in runtime/local.json and Capri/configs/local.json. Model-specific adapters belong in model1/2/3/src. Original protocol copies and historical evidence remain unchanged. New model runs go under the appropriate modelN/runs/<descriptive-name>; generator runs under Capri/runs. Historical scripts need path/protocol adaptation before launching.

Preserve the pixel-only target workflow and custom/DINO/mixed controller mixture. No VLMs, human target strings or manually chosen family lengths enter generation. Reuse A/B; do not regenerate or duplicate datasets inside reader folders. Model1-B ranks A; Model1-A ranks B. Model2 keeps that ranking fixed; Model3 starts there and adapts using synthetic training labels. One run per model/dataset combination, not repeated seed trials.

Both Model1 runs completed20epochs and the coordinator stopped as requested. Migration is not authorization to start Model2/3. Freeze the new curriculum, sampling budget and selection protocol before launching. Do not silently use the old95k/5k split or historical three-run launchers. Match the100k/20epoch Model1 budget; adaptive rescoring overhead must be reported separately.

Real DarkBay is a repeatedly inspected historical test, not untouched validation. Freeze predictions before scoring; never use real labels to steer curriculum or checkpoint selection. For shoot-rate policies, learn confusions and attempt/skip rules on synthetic validation, freeze policy and real decisions, then score real labels. Clearly distinguish historical real confidence sweeps. Requested diagnostics do not change training.

Do not delete original source experiments or model files without explicit authorization. Report what was verified and remaining dependencies honestly. Source-map links can contain old C: strings as immutable provenance; current execution paths and write destinations must use E:.
''')
    doc('CLAUDE.md',r'''# Start here

Read README.md, AGENTS.md, then handoff/CURRENT_STATE.md. This is the shared thesis workspace; all work and caches must stay on E:. handoff/SOURCE_MAP.json resolves original experiment paths and handoff/VALIDATION.json records migration checks. Model1-A/B completed20epochs. New Model2/3 A/B runners and rankings remain pending. No training was launched by migration; preserve the stop-after-Model1-B instruction until the user authorizes further training. Do not launch copied historical scripts unchanged or infer the current state from old protocol prose.
''')
    doc('handoff/CURRENT_STATE.md',r'''# Current state

Migration: requested12November2026, copy-first into E:/pranav/Thesis-Pranav-Final. See COPY_MANIFEST.jsonl and VALIDATION.json for hashes and checks. Originals remain available at E:/pranav/CAPRI. The final workspace is the new place for thesis organization and future implementation.

Completed: CAPRI joint20 training;100k A and100k B generation/audits; standard PARSeq Model1-A and1-B training for20epochs. A final615/900=68.33%; B573/900=63.67% on the same900 historical real images. At epoch15 A592/900 and B548/900; B epoch17 was572/900. These diagnostics led the author to retain the existing20epoch schedule; they are not independent validation.

Original coordinator ended intentionally on its STOP marker after B. A historical status saying failed means requested stop, not failed training. No new training has started. Explicit request remains stop after Model1-B; subsequent evaluation and migration were separately authorized.

Next work: adapt existing static/adaptive implementations to these two corpora; compute Model1-B ranking A and Model1-A ranking B; freeze rankings and curriculum rules; verify sampler, checkpoint/resume and image budgets; only launch on authorization. Four runs remain: Model2-A/B and Model3-A/B. Rank once per direction and share the initial ranking between Model2 and3. Static rankings stay fixed. Adaptive begins easy-only then rescans the training pool, smooths ranks and changes bucket sampling. See model3/adaptive_curriculum.md.

Recommend fixed final20epoch selection with all100k optimization images, matching Model1. This remains a proposed curriculum protocol, not a configured run. Do not silently inherit the older95k training/5k validation allocation or add extra datasets. No real-test threshold tuning.

Copied model1/src/OpenOCR and Capri dependencies preserve the actual code; the existing E: Python environment remains external. Historical entrypoints have path assumptions. Current relative manifests and relocated Model1 configs describe the new files. Shared source is Capri/src/capri_controller. Migration smoke verification proves reader weights/data can be consumed from the new root, not that unimplemented curriculum runners are ready.
''')
    doc('Capri/README.md',r'''# CAPRI

The selected thesis generator is models/capri/checkpoint.pt (epoch20 EMA). It is the lightweight3.66M core plus fine-tuned DINO font branch and independent count transformer. Generation uses internal count and trained font outputs; the separate counter is not used in the renderer flow.

models/controller_dependencies contains custom encoder, text mask, original DINO controller weights, and a legacy hybrid needed for original calibration. That hybrid is a calibration dependency, not a second thesis baseline. The historical generic pipeline resolves that hybrid name; therefore do not run the copied capri.py generation/training commands without adapting them to selected_generator and the new paths. This migration preserves source behavior rather than silently routing to the wrong model.

See models/registry.json, architecture.md, training_setup.md and baseline_selection.md. Original source/config snapshots are in handoff/original_configuration and runs/generator_training. Shared reader curriculum implementations remain under src/capri_controller/training. Preserve their historical protocol provenance; create new A/B adapters before training.
''')
    doc('Capri/architecture.md','''# Architecture

Original lightweight CAPRI core (~3.66M parameters) learns image/background/appearance attributes and internal character count. A fine-tuned DINOv2 ViT-S/14 register branch adds font attribute residuals and a128D font embedding. An independent four-layer128-wide count transformer is trained jointly. The generator consumes pixels only. Three custom/DINO/mixed controllers refine recipes. Current generation uses CAPRI internal count, with a unique leading model-predicted count supported by at least50/100 references, and uses the learned font branch. It does not use the separate counter at generation time.

Exact architecture, losses and limitations: runs/generator_training/protocol.json and src/capri_controller/lightweight. Missing text-ordinal and font width/slant supervision remains a limitation; improvement cannot be attributed solely to font intelligence without ablation.
''')
    doc('Capri/training_setup.md','''# Training setup

Completed joint training:100000 synthetic scenes,20epochs, original lightweight core plus fine-tuned DINO font branch plus separate count transformer. Final epoch20 EMA selected. Logical batch128 with32-image microbatches and pixel-budget accumulation; AdamW base LR0.0005, DINO LR0.000005, weight decay0.05, warmup300steps, cosine-style original schedule and minimum LR0.00001, EMA0.999, BF16. Consult the saved protocol for exact behavior rather than reconstructing from this summary.

Training scenes and font-reference metadata: training_data/scenes_and_font_references. Actual prepared training cache: training_data/prepared_cache. Frozen source snapshots, metrics and checkpoint history: runs/generator_training. Final checkpoint lives once in models/capri. Original paths in cache metadata are provenance; future training must use a relocated adapter and fresh output directory.
''')
    doc('Capri/baseline_selection.md','''# Selected thesis baseline

The author specified one CAPRI baseline and preferred the newer generator if it beat the previous one. The completed matched Model1-A reader comparison supports selecting the new joint20 checkpoint for this thesis workspace:615/900 (68.33%) versus the previous generator reader492/900 (54.67%), same single-run initialization setting and final20epochs on100k generated images. B is separately573/900 (63.67%); A was the predesignated comparison, not selected as the better A/B result.

Checkpoint SHA256:28f95fd8daf3831fb860ab59e15b8b65bd4a68bacc434eb7eb7a296e6190b764. The original workspace registry is preserved unchanged; the final workspace selected_generator record identifies this baseline. The copied generic old pipeline is not automatically adapted to it.

This is evidence on one repeatedly inspected DarkBay test set. It does not establish a font-only causal improvement or superiority on every scheme. Current reader evaluation used CPU FP32; the historical comparison used GPU FP32. Original comparison and protocols are retained in model1/results and handoff/study_records.
''')
    doc('model1/training_setup.md','''# Model 1: standard PARSeq

Completed one reader on A and one on B, each100000 synthetic images and20epochs. SVTRv2LNConvTwo33 encoder with PARSeq decoder, scratch initialization, batch128, AdamW LR0.00035, weight decay0.05, OneCycleLR with2warmup epochs, BF16 training. Fixed final epoch20 selection, no validation or real-score selection. Saved seed48 is a reproducibility value, not a plan for repeated trials.

Exact historical configs: configs/darkbay/A,B/original.yml. Relocated configs point at shared datasets and fresh reproduction destinations; they are not instructions to retrain. Final checkpoints: models/parseq/darkbay/A,B. Other saved epochs and training logs: runs/darkbay/A,B. Results include requested15/17/20epoch diagnostics and older experiments explicitly under historical/.

Real evaluation:900 RGB-hash-verified historical DarkBay images excluding100 controller inputs; RGB bicubic32x128, normalization0.5, saved autoregressive decoding with one refinement, no forced length or narrowed charset. Freeze predictions before labels. A615/900; B573/900. Use src/verify_migrated_readers.py to validate migrated inference against frozen predictions without scoring labels.
''')
    for model in ('model2','model3'):
        doc(model+'/training_setup.md',f'''# {model}: new A/B runs pending

Reuse datasets/darkbay/A and B. Exactly one20epoch run per direction is planned; no training has started. Model1-B supplies the initial A ranking; Model1-A supplies B ranking. The two100k corpora are already complete. Reuse the original Model1 architecture, initialization, optimizer, preprocessing and optimization budget unless a protocol explicitly records a change.

The older implementation and results exist but assume95k/5k split, old data paths and repeated runs. Do not run it unchanged. Finalize the new sampling and selection protocol and create verified adapters before launch. Recommended checkpoint rule: fixed final epoch20 on100k optimization images, matching Model1. Synthetic validation design remains unresolved if needed for any additional policy; real labels must never fill that role.

Canonical shared code is under Capri/src/capri_controller/training. Historical evidence under results/historical is not a completed new A/B experiment. New checkpoints belong in models/parseq/darkbay/A,B and new logs/configs in runs/.
''')
        doc(model+'/src/README.md',f'# Shared implementation\n\nUse ../../Capri/src/capri_controller/training/{"ds2.py and ds2_data.py" if model=="model2" else "ds3.py, ds3_data.py and ds3_checks.py"}. These preserved historical implementations require new A/B adapters. Avoid duplicate common code. No launch-ready new runner exists yet.')
    doc('model2/curriculum.md','''# Static curriculum

One teacher scores the opposite dataset using its synthetic answers. Freeze each image difficulty and easy/medium/hard membership before the student's training. Model2 changes the sampling mixture according to a predetermined schedule; the ranking itself stays fixed.

Historical schedule: epoch1 easy-only; epoch2 50%easy/50%medium; epochs3-20 15%easy/25%medium/60%hard. Historical scoring used a three-reader ensemble and95k/5k split. New A/B runs will use one opposite-dataset teacher and100k optimization images; adaptation and final schedule confirmation are pending.

Store each new initial ranking once under rankings/darkbay/A or B. Model3 references these same files rather than duplicating them. Real labels play no role in difficulty.
''')
    doc('model3/adaptive_curriculum.md','''# Adaptive curriculum

Historical implemented rule: initialize from the teacher's ranking and train epoch1 on easy images only. After each epoch, rescore every synthetic training image using the current reader, with deterministic unaugmented FP32 inference and known synthetic labels. Difficulty is summed negative log probability of correct characters from free autoregressive outputs, converted to percentile ranks.

Smooth difficulty as50%previous smoothed rank plus50%current rank. Rebuild three equally sized easy/medium/hard groups. Estimate each group's remaining difficulty with mean(1-exp(-min(NLL,50))). Allocate at least20% and at most50% of next-epoch draws to each group. Limit subsequent changes to10percentage points per epoch; the initial easy-only transition has a special uniform reference. Sample with replacement. Save ranking, bucket masses, model/optimizer/scheduler and random states for resume. Extra full-pool inference adds compute beyond static training.

New A/B design starts from the opposite Model1 teacher, then Model3 updates its own training-pool ranking. No images are regenerated. Reuse Model2's frozen initial ranking. The previous95k/5k split and three-run orchestration are historical and must be adapted; new100k,20epoch runners and smoke checks remain pending. Original exact code: Capri/src/capri_controller/training/ds3_data.py. Original protocol and actual final-checkpoint selection amendments remain under results/historical.
''')
    doc('handoff/CLAIMS_AND_EVIDENCE.md','''# Claims and evidence

| Claim | Evidence | Limit |
|---|---|---|
| A/B each100k,200k unique RGB images, zero cross overlap | datasets/darkbay/manifests/audit.json | Shared controller recipes; not independent schemes |
| Model1-A final615/900,68.33% | model1/results/darkbay/A/evaluation_darkbay_cpu_requested_v2/summary.json | Historical repeatedly examined test |
| Model1-B final573/900,63.67% | model1/results/darkbay/B/evaluation_darkbay_cpu_epoch20_requested/summary.json | One reader run |
| Later checkpoints improved over15 | handoff/study_records/epoch15_comparison.json and epoch20_comparison.json |15checkpoint is part of20epoch LR schedule |
| New generator chosen as thesis baseline | Capri/baseline_selection.md | No font-only ablation; not proof for every scheme |
| New Model2/3 ready to train | NOT established | Rankings, adapters, protocol and checks pending |
| A and B equivalent | NOT established | Similar generation process does not prove equal reader outcomes |
| Migration preserved files | handoff/COPY_MANIFEST.jsonl, VALIDATION.json | External Python runtime still required |
''')
    doc('handoff/DEPENDENCIES.md',f'''# Dependencies and relocation limits

Existing Python executable: {cfg['python']}. Do not move a Windows virtual environment by copying it. It may rely on a system Python base installation for read-only interpreter files. All project writes, caches and temporary outputs are set to E: by environment.ps1. Package inventory is recorded separately during validation.

Copied: OpenOCR source and PARSeq overlay under model1/src; controller/decoder code including DINO vendor source under Capri/src; renderer and assets under Capri/renderer; sourcefree_synth, feature_bank, resources, font files and licenses under Capri/dependencies/sourcefree_synth. Original licenses are retained. Copied immutable catalogs can contain original paths; SOURCE_MAP.json resolves source files. Historical code is not a new portable runner by itself.

Validated scope is file integrity, shared data access and loaded-checkpoint reader inference from the new root. New generator orchestration and static/adaptive training adapters are still pending; do not claim an end-to-end new training launch has passed. For another computer, recreate dependencies from runtime/package_inventory.json, update runtime/local.json and regenerate machine-local configs. No original experiment is deleted or moved out from under historical scripts.
''')
    doc('handoff/CHANGELOG.md','''# Change log

2026-11-12: Created requested thesis workspace by verified copying, not destructive relocation. Organized one CAPRI baseline, shared A/B datasets, completed Model1 checkpoints/results, historical Model2/3 implementations/results and explicit pending new curricula. Preserved original source/config snapshots and historical paths. Added relative manifests, relocated Model1 configs, E-drive environment bootstrap, provenance map and migration verification tools. Original E:/pranav/CAPRI remains intact. No training launched. See COPY_MANIFEST.jsonl and VALIDATION.json for actual verification and limitations.
''')
    doc('environment.ps1',r'''$thesisRoot = $PSScriptRoot
$thesisConfig = Get-Content (Join-Path $thesisRoot 'runtime/local.json') -Raw | ConvertFrom-Json
$thesisPython = $thesisConfig.python
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = Join-Path $thesisRoot 'runtime/temporary'
$env:TMP = $env:TEMP
$env:TORCH_HOME = Join-Path $thesisRoot 'runtime/cache/torch'
$env:HF_HOME = Join-Path $thesisRoot 'runtime/cache/huggingface'
$env:XDG_CACHE_HOME = Join-Path $thesisRoot 'runtime/cache'
$env:MPLCONFIGDIR = Join-Path $thesisRoot 'runtime/cache/matplotlib'
$env:PIP_CACHE_DIR = Join-Path $thesisRoot 'runtime/cache/pip'
$env:NO_ALBUMENTATIONS_UPDATE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:CAPRI_CONFIG = Join-Path $thesisRoot 'Capri/configs/local.json'
$env:CAPRI_PROJECT = Join-Path $thesisRoot 'Capri/dependencies/sourcefree_synth'
$env:CAPRI_RENDERER = Join-Path $thesisRoot 'Capri/renderer'
$env:PYTHONPATH = ((Join-Path $thesisRoot 'model1/src/parseq_packages'), (Join-Path $thesisRoot 'model1/src/OpenOCR'), (Join-Path $thesisRoot 'Capri/src')) -join ';'
New-Item -ItemType Directory -Force $env:TEMP,$env:TORCH_HOME,$env:HF_HOME,$env:XDG_CACHE_HOME,$env:MPLCONFIGDIR,$env:PIP_CACHE_DIR | Out-Null
''')
    shutil.copy2(Path(__file__),R/'handoff/finalization_script.py')
    shutil.copy2(Path(__file__).with_name('thesis_workspace_cli.py'),R/'workspace.py')
    shutil.copy2(Path(__file__).with_name('verify_migrated_readers.py'),R/'model1/src/verify_migrated_readers.py')
    shutil.copy2(Path(__file__).with_name('complete_thesis_validation_20261112.py'),R/'handoff/validation_script.py')
    print(json.dumps(dict(documentation_ready=True,copied_files=len(records),bytes=sum(r['bytes'] for r in records),source_map='handoff/SOURCE_MAP.json')),flush=True)

if __name__=='__main__':main()
