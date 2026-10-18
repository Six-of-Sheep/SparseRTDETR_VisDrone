"""GT-free fixed sixth view; original NMS and original C remain unchanged."""
from dataclasses import dataclass
from slice_core import (Candidate, _box64, _finite, _integer, merge_candidates,
                        select_view_candidates, tile_windows)

VERSION='r18-fixed-sixth-view-v1'
VARIANTS=('C-center','C-flip')
EXTRA_VIEW_ID=5


@dataclass(frozen=True)
class SixthCandidate(Candidate):
    """Versioned extension of Candidate allowing view5; all other rules identical."""
    def __post_init__(self):
        box=_box64(self.box_xyxy);score=_finite(self.score,'score')
        if not 0<=score<=1:raise ValueError('sigmoid probability outside [0,1]')
        if len(self.input_shape)!=2:raise ValueError('input_shape')
        shape=tuple(_integer(v,'shape dimension',1) for v in self.input_shape)
        for name,minimum in [('view_id',0),('query_id',0),('category_id',1),('view_candidate_rank',1)]:
            _integer(getattr(self,name),name,minimum)
        if self.view_id>5 or self.query_id>=shape[0] or self.category_id>shape[1] or self.view_candidate_rank>shape[0]*shape[1]:
            raise ValueError('sixth-view identity outside input shape')
        object.__setattr__(self,'box_xyxy',box);object.__setattr__(self,'score',score)
        object.__setattr__(self,'input_shape',shape)


def center_window(height,width):
    _integer(height,'height',1);_integer(width,'width',1)
    h=(5*height+8)//9;w=(5*width+8)//9
    x=(width-w)//2;y=(height-h)//2
    if (h,w)!=(tile_windows(height,width)[0].height,tile_windows(height,width)[0].width):
        raise ValueError('center size differs from original corner tiles')
    return (x,y,x+w,y+h)


def map_extra_xyxy(boxes,height,width,variant):
    if variant not in VARIANTS:raise ValueError('only frozen center/flip supported')
    _integer(height,'height',1);_integer(width,'width',1)
    crop=center_window(height,width)
    result=[]
    for box in boxes:
        x1,y1,x2,y2=_box64(box)
        mapped=(x1+crop[0],y1+crop[1],x2+crop[0],y2+crop[1]) if variant=='C-center' else (width-x2,y1,width-x1,y2)
        result.append(_box64(mapped))
    return result


def extra_candidates(mapped_boxes,probabilities):
    # Per-view top600 ranking has no dependence on the view ID. Select with the
    # unchanged local selector, then attach the actual versioned identity5.
    local=select_view_candidates(mapped_boxes,probabilities,view_id=0)
    return [SixthCandidate(**(p.as_dict()|{'view_id':EXTRA_VIEW_ID})) for p in local]


def merge_six(base,extra=None,*,variant=None):
    base=list(base)
    if any(p.view_id not in range(5) or type(p) is not Candidate for p in base):
        raise ValueError('base is not the original five-view candidate pool')
    if variant is None:
        if extra is not None:raise ValueError('disabled capability received extra candidates')
        return merge_candidates(base)  # Literal original C path, no rewriting.
    if variant not in VARIANTS or extra is None:raise ValueError('sixth view incomplete')
    if len(base)!=3000 or any(sum(p.view_id==v for p in base)!=600 for v in range(5)) or any(p.input_shape!=(300,10) for p in base):
        raise ValueError('six-view execution requires original five complete top600 pools')
    extra=list(extra)
    if len(extra)!=600 or any(type(p) is not SixthCandidate or p.view_id!=5 for p in extra):
        raise ValueError('extra view must supply original top600 with explicit view5')
    # Original classwise NMS accepts Candidate subclasses; same arithmetic,
    # threshold, score/identity ordering and output budgets, no coordinate fusion.
    result=merge_candidates(base+extra)
    return result|{'sixth_view_version':VERSION,'variant':variant,'extra_view_id':5}
