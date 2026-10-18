"""Synthetic checks using the original raw-ignore implementation, never a replica."""
def run():
    import boxvote_job as j
    from slice_core import Candidate
    from boxvote_core import coordinate_vote
    from sparse_rtdetr.localization.data import parse_raw
    from sparse_rtdetr.r0.evaluation import bridge_inputs
    row={'stable_image_id':'filter-only-synthetic','width':100,'height':100,'split':'val'}
    def c(q,box,score):return Candidate(box,score,0,q,1,q+1)
    pool=[c(0,(0.,0.,10.,10.),1.),c(1,(2.,0.,12.,10.),.5),c(2,(50.,50.,60.,60.),.4)]
    voted=coordinate_vote(pool)['all']
    def count(raw,ps):
        parsed=parse_raw(raw,row['stable_image_id'])
        return bridge_inputs([row],[parsed],[j.prediction(row['stable_image_id'],p) for p in ps])[2]
    none=count(b'',voted[:1]);j.require(none['detections_before']==none['detections_after']==1,'no filtering differs')
    all_=count(b'0,0,100,100,0,0,0,0\n',voted[:1]);j.require(all_['detections_before']==1 and all_['detections_after']==0,'all filtering differs')
    partial=count(b'0,0,20,20,0,0,0,0\n',voted);j.require(partial['detections_before']==2 and partial['detections_after']==1,'partial filtering differs')
    # Exactly-half original drops, moved vote survives. Counts may legally change.
    ignored=b'0,0,5,10,0,0,0,0\n'
    before=count(ignored,pool[:1]);after=count(ignored,voted[:1])
    j.require(before['detections_after']==0 and after['detections_after']==1,'vote coordinates not used for ignore')
    # Budget first. Lower ranked valid prediction may not refill dropped top1.
    truncated=count(b'0,0,20,20,0,0,0,0\n',voted[:1])
    j.require(truncated['detections_before']==1 and truncated['detections_after']==0,'truncation/no refill differs')
    return {'status':'PASS','no_filter':none,'all_filter':all_,'partial_filter':partial,
            'original_top1':before,'voted_top1':after,'budget_before_ignore_no_refill':truncated,
            'GT_used_in_postprocess':False,'filter_implementation':'original bridge_inputs'}
