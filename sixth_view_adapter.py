"""CPU-testable pixel/decode interfaces; no detector construction or dispatch."""
from sixth_view_core import center_window,map_extra_xyxy,extra_candidates,VARIANTS


def make_six_pixels(data,rows,variant):
    if variant not in VARIANTS:raise ValueError('unknown sixth view')
    from infer_seed import load_original,pixels
    from slice_core import tile_windows
    from PIL import Image
    views=[[] for _ in range(6)];geometry=[]
    for row in rows:
        image=load_original(data,row)
        try:
            views[0].append(pixels(data,image))
            tiles=tile_windows(image.height,image.width)
            for tile in tiles:
                crop=image.crop(tile.crop_box)
                try:views[tile.view_id].append(pixels(data,crop))
                finally:crop.close()
            window=center_window(image.height,image.width)
            extra=image.crop(window) if variant=='C-center' else image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            try:views[5].append(pixels(data,extra))
            finally:extra.close()
            geometry.append({'original_tiles':[t.as_dict() for t in tiles],
                             'extra':{'view_id':5,'variant':variant,'crop_half_open':list(window) if variant=='C-center' else None,
                                      'flip':'original RGB before certified resize' if variant=='C-flip' else None}})
        finally:image.close()
    return views,geometry


def decode_extra(logits,boxes,row,variant):
    import torch
    from sparse_rtdetr.localization.consumer_geometry import corners
    if variant not in VARIANTS or logits.shape!=(300,10) or boxes.shape!=(300,4):raise ValueError('extra output schema')
    if logits.dtype!=torch.float32 or boxes.dtype!=torch.float32 or logits.device.type!='cpu' or boxes.device.type!='cpu':raise ValueError('extra decode requires original CPU FP32 outputs')
    if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all() or (boxes[:,2:]<=0).any():raise ValueError('extra nonfinite or nonpositive output')
    height,width=row['height'],row['width'];crop=center_window(height,width)
    h,w=(crop[3]-crop[1],crop[2]-crop[0]) if variant=='C-center' else (height,width)
    xy=torch.cat((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),-1)
    corners('sixth-view/normalized',xy)
    xy*=torch.tensor([w,h]*2,dtype=torch.float32)
    corners('sixth-view/local-pixels',xy)
    mapped=map_extra_xyxy(xy.tolist(),height,width,variant)
    return extra_candidates(mapped,logits.sigmoid().numpy())
