"""Local epoch clock, using the original raw reader and augmentation operators."""
import torch
from ..localization.data import RawAugmentation, training_loader
from ..protocol_r18.data import BoundDataset, input_record
from ..r0.evaluation_data import Development548


class _NoStrongOperation:
    actual_params = None

    def __call__(self, image, target):
        return image, target


class LocalAugmentation:
    """Skip only photo/zoom/crop at local e22; retain local RNG seed/hflip.

    RawAugmentation's epoch also seeds each sample. Mapping e22 to e118 would
    silently change that seed. Identity strong ops retain the original local
    epoch seed and the native weak geometry/sanitize/flip/resize sequence.
    """
    def __init__(self, hw=(768, 1344)):
        self.strong = RawAugmentation(hw)
        self.weak = RawAugmentation(hw)
        self.weak.photo = _NoStrongOperation()
        self.weak.zoom = _NoStrongOperation()
        self.weak.crop = _NoStrongOperation()

    def __call__(self, image, raw, *, epoch, **kw):
        if not 1 <= epoch <= 24:
            raise ValueError('local augmentation epoch')
        im, target, trace = (self.strong if epoch <= 21 else self.weak)(image, raw, epoch=epoch, **kw)
        trace['augmentation_strong'] = kw.get('training', True) and epoch <= 21
        trace['local_augmentation_stop_epoch'] = 22
        return im, target, trace


class Dataset(BoundDataset):
    def __init__(self, spec):
        r = spec['roots']
        super().__init__(role='train_core', manifest_path=spec['manifest']['path'],
                         images_path=r['images'], annotations_path=r['annotations'], seed=0, root_binding=r)
        self.transform = LocalAugmentation()


def windows(data, epoch):
    for im, targets, traces in training_loader(data, epoch=epoch, completed_windows=0):
        identity = input_record(im, targets, traces)
        identity.pop('spawn_workers')
        yield im, targets, identity


class Fixture:
    """Constructed images and GT only; never a real data or weight admission."""
    def __init__(self):
        self.records = [dict(stable_image_id='SYNTHETIC_NOT_SCIENTIFIC/dev/%03d' % i,
                             width=160, height=128) for i in range(6)]
        self.transform = RawAugmentation((128, 128))

    def raw(self, row):
        from ..localization.data import parse_raw
        return parse_raw(b'12,16,24,20,1,1,0,0\n60,40,40,40,1,2,0,0\n110,20,40,70,1,3,0,0\n0,0,8,8,0,0,0,0\n', row['stable_image_id'], expected_rows=4)

    def item(self, row, position):
        from PIL import Image
        im = Image.new('RGB', (160, 128), (30 + position, 50, 80))
        try:
            return self.transform(im, self.raw(row), epoch=120, seed=0, position=position,
                                  image_id=row['stable_image_id'], training=False)[:2]
        finally:
            im.close()

    def window(self, epoch):
        rows = self.records[:2]
        items = [self.item(r, epoch + i) for i, r in enumerate(rows)]
        im = torch.stack([x[0] for x in items]); targets = [x[1] for x in items]
        from ..protocol_r18.engine import output_digest
        return im, targets, {'epoch': epoch, 'ids': [r['stable_image_id'] for r in rows],
                             'input_SHA': output_digest([im, targets])}


def development(c):
    if c['mode'] == 'CPU_SYNTHETIC_SABL_FORMAL':
        d = Fixture()
    else:
        spec = c['metadata']['data']['development']
        d = Development548({'manifest': spec['manifest']['path'], 'roots': spec['roots']})
    return d.records, d
