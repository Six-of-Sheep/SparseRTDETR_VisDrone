import copy
import unittest
from sixth_view_admission import adapt_resources


class AdmissionTests(unittest.TestCase):
    def fixture(self):
        return dict(runtime_root='/runtime', boot_id='boot', gpu_uuid='GPU-test',
                    storage_identity=dict(canonical_mount='/disk', platform_entry='/disk',
                        mount={'type': 'ext4'}, st_dev=1, planning_cap_bytes=100),
                    environment=dict(torch='2.4.1', torchvision='0.19.1', cuda_build='12.4', cudnn=90100))

    def test_explicit_mapping_non_mutating(self):
        c = self.fixture(); before = copy.deepcopy(c)
        mapped = adapt_resources(c)
        self.assertEqual(mapped['storage']['identity'], c['storage_identity'])
        mapped['storage']['identity']['st_dev'] = 9
        self.assertEqual(c, before)

    def test_conflict_rejected_and_consistent_preserved(self):
        c = self.fixture(); c['storage'] = {'identity': copy.deepcopy(c['storage_identity']), 'other': 'preserve'}
        self.assertEqual(adapt_resources(c), c)
        c['storage']['identity']['st_dev'] = 2
        with self.assertRaisesRegex(RuntimeError, 'conflicting'): adapt_resources(c)

    def test_missing_contract_fields_rejected(self):
        for key in self.fixture():
            with self.subTest(key=key):
                c = self.fixture(); del c[key]
                with self.assertRaisesRegex(RuntimeError, 'missing'): adapt_resources(c)

    def test_incomplete_nested_fields_rejected(self):
        for parent in ('storage_identity', 'environment'):
            for key in self.fixture()[parent]:
                with self.subTest(parent=parent, key=key):
                    c = self.fixture(); del c[parent][key]
                    with self.assertRaises(RuntimeError): adapt_resources(c)
        c = self.fixture(); c['storage'] = {}
        with self.assertRaises(RuntimeError): adapt_resources(c)


if __name__ == '__main__': unittest.main()
