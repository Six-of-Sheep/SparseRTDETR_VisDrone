"""Exact metadata roles admitted before reading bytes (no data/model imports)."""
from pathlib import Path
from .runio import REMOTE_ROOT,SELECTION_SHA,require

WORKSPACE=Path('C:/Users/liu/Documents/Lzy_CV')
EXPERIMENT=WORKSPACE/'workspace_management/delivery/experiments'
LOCAL_STAGE=EXPERIMENT/'localization-distribution/f0-seed0-e120-20261009-001'
MANIFESTS={
    'train_core':('df87e6a377a5546d3d3fe0cd0e860d15c2fe5e8003ea87d887153579999a4e9d',4869,'train'),
    'development':('5f4d788591b60bd2a6223c15e548248f4c0296656373d6d98c016d912a7537dd',548,'val'),
}
LOCAL_MANIFESTS={
    'train_core':EXPERIMENT/'c2-local-refinement/local-preflight-001/assets/train_core_manifest.json',
    'development':EXPERIMENT/'c2-local-refinement/autodl-migration-001/cloud-formal-20261008-001/materials/legacy/SparseRTDETR_VisDrone/artifacts/data/visdrone_protocol_v2_conversion_r3/development_manifest.json',
}

def metadata_path(kind,path,expected_sha):
    p=Path(path)
    require(p.is_absolute(),'metadata path must be absolute')
    for candidate in (p,p.resolve()):
        parts=[s.lower().replace('-','_') for s in candidate.parts]
        require(not any(any(k in s for k in ('holdout','confirmatory','test_dev','testdev')) for s in parts),'forbidden metadata path before read')
    require(not any(x.is_symlink() for x in (p,*p.parents)),'metadata path traverses symlink')
    if kind in MANIFESTS:
        require(expected_sha==MANIFESTS[kind][0],'metadata role digest is not the frozen digest')
        choices=(LOCAL_MANIFESTS[kind],Path(REMOTE_ROOT)/'materials'/(kind+'_manifest.json'),Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution/f0-e026-labdiag-20261009-001/materials')/(kind+'_manifest.json'),Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution/f0-e026-labdiag-20261009-001/numerical-repair-001/materials')/(kind+'_manifest.json'))
    else:
        require(kind=='selection' and expected_sha==SELECTION_SHA,'unknown metadata role or digest')
        choices=(LOCAL_STAGE/'trace-selection.json',Path(REMOTE_ROOT)/'materials/trace-selection.json',Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution/f0-e026-labdiag-20261009-001/materials/trace-selection.json'),Path('/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/localization-distribution/f0-e026-labdiag-20261009-001/numerical-repair-001/materials/trace-selection.json'))
    require(p.resolve() in {q.resolve() for q in choices if q.is_absolute()},'metadata path is not an admitted role source')
    return p.resolve()
