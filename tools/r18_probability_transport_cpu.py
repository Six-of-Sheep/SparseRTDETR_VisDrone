"""Bounded synthetic CPU arithmetic. Never imports a detector or reads datasets.

Run on Linux/WSL with CPU PyTorch: python -B tools/r18_probability_transport_cpu.py OUT
OUT must not exist. The JSON keeps failed assertions and counterexamples.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

START = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
DEFINITION = ROOT / 'docs/research/r18_reference_topic/CPU_DEFINITION.json'
CFG = json.loads(DEFINITION.read_text(encoding='utf-8'))
resource.setrlimit(resource.RLIMIT_AS, (CFG['address_space_bytes'], CFG['address_space_bytes']))
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
signal.alarm(CFG['wall_seconds'])
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64)
EPS = CFG['epsilon']


def tensor(x, grad=False):
    return torch.tensor(x, dtype=torch.float64, requires_grad=grad)


def weights():
    # Same fixed nonuniform support as pinned D-FINE, not uniform bins.
    m, scale, up = CFG['reg_max'], CFG['reg_scale'], CFG['up']
    step = (up * scale + 1) ** (2 / (m - 2))
    return tensor([-2 * up * scale] + [1 - step ** i for i in range(m // 2 - 1, 0, -1)]
                  + [0.] + [step ** i - 1 for i in range(1, m // 2)] + [2 * up * scale])


def grids(box):
    """box=(cx,cy,w,h) in one common continuous image coordinate frame.

    Rows L,T,R,B; columns preserve native increasing W index. L/T decrease
    physically; R/B increase. No image clipping or coordinate gradients.
    """
    b = box.detach()
    if b.shape != (4,) or not bool(torch.isfinite(b).all()) or bool((b[2:] <= 0).any()):
        raise ValueError('invalid reference box')
    cx, cy, w, h = b
    d = .5 + weights() / CFG['reg_scale']
    return torch.stack((cx-w*d, cy-h*d, cx+w*d, cy+h*d))


def matrix(old, new):
    """Column stochastic T[new,old], detached geometry, native index output."""
    x, y = old.detach(), new.detach()
    if x.ndim != 1 or y.ndim != 1 or len(y) < 2 or not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise ValueError('invalid grid')
    ys, order = torch.sort(y)
    if not bool((ys[1:] > ys[:-1]).all()):
        raise ValueError('duplicate grid coordinate')
    t = torch.zeros((len(y), len(x)))
    for n, v in enumerate(x):
        if v <= ys[0]:
            t[order[0], n] = 1.
        elif v >= ys[-1]:
            t[order[-1], n] = 1.
        else:
            hi = int(torch.searchsorted(ys, v))
            a = (v-ys[hi-1]) / (ys[hi]-ys[hi-1])
            t[order[hi-1], n], t[order[hi], n] = 1-a, a
    return t


def transport(p, old, new):
    if p.shape != old.shape or not bool(torch.isfinite(p).all()) or bool((p < 0).any()) or abs(float(p.sum().detach())-1) > 1e-10:
        raise ValueError('invalid probability')
    t = matrix(old, new)
    out = t @ p
    x, y = old.detach(), new.detach()
    outside = (x < y.min()) | (x > y.max())
    return out, {'outside_mass': float(p.detach()[outside].sum()),
                 'signed_moment_bias': float((out.detach()*y).sum()-(p.detach()*x).sum())}


def update(p, old, new, residual):
    q, receipt = transport(p, old, new)
    return torch.softmax(torch.log(q + EPS) + residual, -1), receipt


def decode(p, g):
    edges = (p * g.detach()).sum(-1)
    l, t, r, b = edges
    if not bool(torch.isfinite(edges).all()) or float((r-l).detach()) <= 0 or float((b-t).detach()) <= 0:
        raise ValueError('invalid decoded box; no repair')
    return torch.stack(((l+r)/2, (t+b)/2, r-l, b-t))


def target_distribution(gt_coordinate, grid):
    """Frozen two-neighbor target in the CURRENT physical coordinate frame."""
    return matrix(gt_coordinate.detach().reshape(1), grid)[:, 0]


def chain(logits, initial, arm):
    if arm not in ('C0', 'C1', 'M') or logits.shape != (3, 4, 33):
        raise ValueError('invalid arm or logits')
    g = grids(initial)
    p = torch.softmax(logits[0], -1)
    b = decode(p, g)
    states = [(p, g, b)]
    z = logits[0]
    for i in (1, 2):
        ng = grids(initial if arm == 'C0' else b)
        if arm == 'C0':
            z = z + logits[i]
            np = torch.softmax(z, -1)
        elif arm == 'C1':
            np = torch.softmax(logits[i], -1)
        else:
            np = torch.stack([update(p[e], g[e], ng[e], logits[i,e])[0] for e in range(4)])
        b = decode(np, ng)
        p, g = np, ng
        states.append((p, g, b))
    return states


def select_ids(ids):
    if len(ids) != len(set(ids)) or len(ids) < 192 or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError('canonical unique IDs required')
    ranked = sorted(ids, key=lambda s: (hashlib.sha256(('PR25-probe-v1\0'+s).encode()).digest(), s.encode()))
    return ranked[:128], ranked[128:192]


def main(path):
    # Exclusive output: failed evidence can never be overwritten by rerun.
    with Path(path).open('x', encoding='utf-8') as stream:
        result = {'scope': 'SYNTHETIC_NOT_SCIENTIFIC', 'definition_sha256': hashlib.sha256(DEFINITION.read_bytes()).hexdigest(),
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'checks': [], 'counterexamples': {}}
        def check(name, ok, details=None):
            result['checks'].append({'name': name, 'pass': bool(ok), 'details': details})
        def near(a,b,tol=1e-10):
            return bool(torch.allclose(a,b,atol=tol,rtol=0))
        try:
            g = grids(tensor([.5,.4,.2,.1], True))
            p = torch.softmax(torch.arange(33.)/8,0)
            check('nonuniform_33_and_four_directions', len(weights()) == 33 and bool((g[:2,1:] < g[:2,:-1]).all()) and bool((g[2:,1:] > g[2:,:-1]).all()) and not near(torch.diff(weights()), torch.diff(weights()).mean().expand(32)))
            check('coordinates_detached', not g.requires_grad)
            for e in range(4):
                t = matrix(g[e],g[e])
                check('identity_matrix_edge_'+str(e), near(t,torch.eye(33)))
                q,r = transport(p,g[e],g[e])
                check('identity_transport_edge_'+str(e), near(q,p))
                updated,_ = update(p,g[e],g[e],torch.zeros(33))
                check('epsilon_formula_edge_'+str(e), near(updated,(p+EPS)/(1+33*EPS)))
            sparse = torch.zeros(33); sparse[0] = 1.
            ep,_ = update(sparse,g[0],g[0],torch.zeros(33))
            result['counterexamples']['epsilon_not_identity'] = {'max_change': float((ep-sparse).abs().max()), 'new_zero_bin_mass': float(ep[1])}
            check('epsilon_is_not_exact_identity', not torch.equal(ep,sparse) and float(ep[1]) > 0)
            for e in range(4):
                ng = grids(tensor([.5,.4,.4,.2]))[e]
                t = matrix(g[e],ng); q,r = transport(p,g[e],ng)
                check('mass_moment_in_domain_'+str(e), near(t.sum(0),torch.ones(33)) and near(q.sum(),tensor(1.)) and near((q*ng).sum(),(p*g[e]).sum()) and r['outside_mass']==0)
                check('affine_covariance_'+str(e), near(t,matrix(g[e]*2.3-7,ng*2.3-7)))
                # Explicit orientation permutation covariance, not nearest-bin indexing.
                check('reverse_index_covariance_'+str(e), near(matrix(g[e].flip(0),ng.flip(0)),t.flip((0,1))))
            x,y = tensor([-2.,0.,3.]),tensor([-1.,.2,1.])
            op = tensor([.2,.3,.5]); q,r = transport(op,x,y)
            result['counterexamples']['outside'] = r
            check('outside_mass_and_signed_bias', abs(r['outside_mass']-.7)<1e-12 and abs(r['signed_moment_bias']+.8)<1e-12)
            a,b = tensor([0.,1.,2.]),tensor([0.,.5,1.5,2.])
            v = tensor([0.,1.,0.]); hist=[]
            for _ in range(4):
                v,_=transport(v,a,b); v,_=transport(v,b,a)
                hist.append({'p':v.tolist(),'mean':float(v@a),'variance':float(v@((a-1)**2))})
            result['counterexamples']['roundtrip_diffusion'] = hist
            check('roundtrip_is_lossy_mean_preserved', not near(v,tensor([0.,1.,0.])) and abs(float(v@a)-1)<1e-12 and hist[-1]['variance']>hist[0]['variance'])
            multi,_=transport(tensor([0.,.5,0.,.5,0.]),tensor([0.,1.,2.,3.,4.]),tensor([0.,2.,4.]))
            result['counterexamples']['two_peaks_to_one'] = multi.tolist()
            check('multimodal_information_lost', near(multi,tensor([.25,.5,.25])))
            v=tensor([0.,0.,1.]); old=tensor([0.,1.,2.]); drift=[]
            for shift in (0.2,0.4,0.6):
                new=tensor([shift,1.,2.-shift]); v,r=transport(v,old,new); drift.append({'mean':float(v@new),**r}); old=new
            result['counterexamples']['repeated_clipping_drift']=drift
            check('repeated_clipping_drift', abs(drift[-1]['mean']-1.4)<1e-12)
            z=tensor([-.3,.2,.4],True); residual=tensor([.1,-.2,.3],True)
            old=tensor([0.,.7,2.],True); new=tensor([-.2,1.,2.2],True)
            utility=tensor([.2,-.7,1.1])
            def objective(zz,rr):
                return update(torch.softmax(zz,0),old,new,rr)[0]@utility
            value=objective(z,residual)
            grads=torch.autograd.grad(value,(z,residual,old,new),allow_unused=True)
            h=CFG['finite_difference_step']; errors=[]
            for which in range(2):
                fd=[]
                for j in range(3):
                    dz=torch.zeros(3);dz[j]=h
                    plus=objective(z.detach()+dz if which==0 else z.detach(),residual.detach()+dz if which==1 else residual.detach())
                    minus=objective(z.detach()-dz if which==0 else z.detach(),residual.detach()-dz if which==1 else residual.detach())
                    fd.append(float((plus-minus)/(2*h)))
                errors.append(float((grads[which]-tensor(fd)).abs().max()))
            check('probability_and_residual_finite_difference', max(errors)<CFG['gradient_tolerance'],errors)
            check('geometry_detach_no_autograd_path',grads[2] is None and grads[3] is None)
            target_gt=tensor(.7,True)
            tg=target_distribution(target_gt,new)
            check('target_barycentric_and_detached',not tg.requires_grad and near(tg.sum(),tensor(1.)) and near(tg@new.detach(),target_gt.detach()))
            endpoint=target_distribution(tensor(9.),new)
            check('outside_target_endpoint',near(endpoint,tensor([0.,0.,1.])))
            const_z=tensor([.1,.2,.3],True)
            const_loss=update(torch.softmax(const_z,0),old,new,torch.zeros(3))[0].sum()
            check('counterexample_constant_objective_zero_gradient',float(torch.autograd.grad(const_loss,const_z)[0].abs().max())<1e-12)
            # An informative prior can help or hurt depending on the target;
            # output changes alone never establish useful information.
            oldp=tensor([.1,.2,.7]); simple=tensor([0.,1.,2.])
            with_prior,_=update(oldp,simple,simple,torch.zeros(3))
            uniform,_=update(torch.ones(3)/3,simple,simple,torch.zeros(3))
            prior_mean=float(with_prior@simple); uniform_mean=float(uniform@simple)
            result['counterexamples']['prior_help_or_harm']={'prior_mean':prior_mean,'uniform_mean':uniform_mean,'gt_help':2.,'gt_harm':0.}
            check('prior_help_and_harm',abs(prior_mean-2)<abs(uniform_mean-2) and abs(prior_mean)>abs(uniform_mean))
            # Stop-gradient finite differences must hold geometry fixed: recomputing
            # reference values is a different numerical function, intentionally not checked.
            initial=tensor([.5,.5,.2,.3],True)
            logits=torch.zeros((3,4,33),requires_grad=True)
            states={arm:chain(logits,initial,arm) for arm in ('C0','C1','M')}
            check('three_arm_zero_init_equal_boxes', all(near(s[-1][2],initial.detach()) for s in states.values()))
            check('same_parameter_count',3*(256+1)*4*33==101772,{'per_arm':101772,'extra_transport_parameters':0})
            for arm,s in states.items():
                loss=s[-1][0][0]@torch.arange(33.)
                gl,gi=torch.autograd.grad(loss,(logits,initial),allow_unused=True,retain_graph=True)
                check('reference_detach_'+arm,gi is None and all(not st[1].requires_grad for st in s))
                check('old_logit_path_'+arm, bool(gl[0].abs().sum()>0) == (arm!='C1'))
            nonzero=(torch.arange(33.)*.02).repeat(3,4,1).requires_grad_()
            changing={arm:chain(nonzero,initial,arm) for arm in ('C0','C1','M')}
            for arm,s in changing.items():
                check('nonzero_reference_rule_'+arm,near(s[1][1],grids(initial if arm=='C0' else s[0][2])) and not near(s[0][2],initial.detach()))
            check('nonzero_reference_changes_C1_M',not near(changing['M'][1][1],changing['M'][0][1]) and not near(changing['C1'][1][1],changing['C1'][0][1]))
            gt=tensor(.43)
            ct0=target_distribution(gt,changing['C0'][1][1][0])
            ct1=target_distribution(gt,changing['C1'][1][1][0])
            check('current_reference_target_not_stale',not near(ct0,ct1) and near(ct0@changing['C0'][1][1][0],gt) and near(ct1@changing['C1'][1][1][0],gt))
            ids=['image-%04d'%i for i in range(300)]
            fit,probe=select_ids(ids)
            check('selection_order_independent_disjoint', (fit,probe)==select_ids(list(reversed(ids))) and not set(fit)&set(probe) and len(fit)==128 and len(probe)==64)
            for name,func in [('duplicate_ids',lambda:select_ids(ids+[ids[0]])),('invalid_reference',lambda:grids(tensor([0.,0.,-1.,1.]))),('duplicate_grid',lambda:matrix(tensor([0.]),tensor([1.,1.]))),('invalid_probability',lambda:transport(tensor([.2,.2]),tensor([0.,1.]),tensor([0.,1.]))),('invalid_decoded_box',lambda:decode(torch.ones((4,33))/33,torch.zeros((4,33))))]:
                try: func(); rejected=False
                except ValueError: rejected=True
                check('reject_'+name,rejected)
            result['status']='PASS' if all(c['pass'] for c in result['checks']) else 'FAIL'
        except Exception as exc:
            result['status']='ERROR'; result['error']=repr(exc)
        result['resources']={'seconds_including_import':time.monotonic()-START,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'affinity':sorted(os.sched_getaffinity(0)),'torch':torch.__version__,'cuda_initialized':torch.cuda.is_initialized()}
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
        print(json.dumps({'status':result['status'],'checks':len(result['checks']),'resources':result['resources']}))
        return 0 if result['status']=='PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
