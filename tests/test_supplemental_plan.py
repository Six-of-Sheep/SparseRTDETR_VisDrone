"""Preparation-only checks; never imports detector or production entrypoints."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'configs/sahi_independent_confirmation.json'
PLAN = ROOT / 'configs/sahi_supplemental_evaluation.json'
EXCLUDED = {'schema', 'task_id', 'qualification_status', 'eligible_confirmation_set',
            'planned_forward_batches_if_1602_qualified', 'this_round', 'final_execution_authorized'}


def check_plan(plan):
    parent = json.loads(PARENT.read_text(encoding='utf-8'))
    expected = {key: value for key, value in parent.items() if key not in EXCLUDED}
    if plan['scientific_protocol'] != expected:
        raise ValueError('scientific protocol changed')
    if plan['parent_independent_plan']['sha256'] != hashlib.sha256(PARENT.read_bytes()).hexdigest():
        raise ValueError('parent SHA conflict')
    if plan['parent_independent_plan']['superseded'] is not False:
        raise ValueError('cannot supersede independent protocol')
    if plan['purpose'] != 'supplemental_evaluation_history_unverified' or plan['task_id'] == parent['task_id']:
        raise ValueError('purpose/identity conflict')
    if plan['independent_qualification'] != 'BLOCKED_PENDING_ACCESS_HISTORY':
        raise ValueError('independent qualification must remain blocked')
    auth = plan['authorization']
    if auth != {'state': 'NOT_GRANTED', 'GPU': False, 'sealed_content': False,
                'scientific_scoring': False, 'legacy_tokens_reusable': False,
                'new_purpose_bound_contract_required': True}:
        raise ValueError('no execution or token authority granted')
    h_ref = plan['human_statements']
    h_path = ROOT / h_ref['path']
    if hashlib.sha256(h_path.read_bytes()).hexdigest() != h_ref['sha256']:
        raise ValueError('human statement SHA conflict')
    human = json.loads(h_path.read_text(encoding='utf-8'))
    if [r['id'] for r in human['items']] != [1, 2, 3] or any(
        r['user_response'] != '用户无法确认' or r['usage_verdict'] != 'unknown'
        for r in human['items']
    ):
        raise ValueError('user unknown cannot be converted to non-use')
    if not {'historical_model_development_use', 'perceptual_near_duplicates', 'flight_site_association'} <= set(plan['claim_policy']['unknowns']):
        raise ValueError('unknowns omitted')
    if 'unseen_independent_confirmation' not in plan['claim_policy']['forbidden']:
        raise ValueError('independence claim cannot be enabled')
    w = plan['workload']
    if w['seeds'] != [0, 1, 2] or w['images'] != 1602 or w['last_batch'] != 2 or w['full_batches_each_seed'] != 401:
        raise ValueError('members/seeds/batch changed')
    if w['all_three_seed_forward_batches_including_warmups'] != 3 * (126 + 401 * 5 + 2 * 5):
        raise ValueError('forward budget conflict')
    if w['image_view_forward_samples_including_warmups'] != 3 * ((64 + 8) * 7 + (1602 + 8) * 5):
        raise ValueError('partial batch sample budget conflict')
    if (w['dual_evaluator_groups'], w['evaluator_calls'], w['training_updates']) != (18, 36, 0):
        raise ValueError('evaluation/update scope changed')
    return True


class SupplementalPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads(PLAN.read_text(encoding='utf-8'))

    def test_actual_plan_and_human_reference(self):
        self.assertTrue(check_plan(self.plan))

    def test_changed_threshold_rejected(self):
        self.plan['scientific_protocol']['C_minus_A_minimum_pp']['600_AP'] = 0.2
        with self.assertRaisesRegex(ValueError, 'scientific'):
            check_plan(self.plan)

    def test_changed_seed_or_identity_rejected(self):
        for key, value in [('purpose', 'independent_confirmation'), ('task_id', 'sahi-independent-confirmation-20261018-001')]:
            p = copy.deepcopy(self.plan); p[key] = value
            with self.assertRaisesRegex(ValueError, 'purpose'):
                check_plan(p)

    def test_qualification_upgrade_rejected(self):
        self.plan['independent_qualification'] = 'PASS'
        with self.assertRaisesRegex(ValueError, 'blocked'):
            check_plan(self.plan)

    def test_execution_and_old_token_reuse_rejected(self):
        for key in ('GPU', 'sealed_content', 'scientific_scoring', 'legacy_tokens_reusable'):
            p = copy.deepcopy(self.plan); p['authorization'][key] = True
            with self.assertRaisesRegex(ValueError, 'authority'):
                check_plan(p)

    def test_parent_or_human_sha_conflict_rejected(self):
        for key in ('parent_independent_plan', 'human_statements'):
            p = copy.deepcopy(self.plan); p[key]['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'SHA'):
                check_plan(p)

    def test_budget_and_seed_omission_rejected(self):
        for key, value in [('seeds', [0]), ('all_three_seed_forward_batches_including_warmups', 6422),
                           ('image_view_forward_samples_including_warmups', 25692), ('evaluator_calls', 18)]:
            p = copy.deepcopy(self.plan); p['workload'][key] = value
            with self.assertRaises(ValueError):
                check_plan(p)

    def test_claim_and_unknown_omission_rejected(self):
        for key in ('forbidden', 'unknowns'):
            p = copy.deepcopy(self.plan); p['claim_policy'][key] = []
            with self.assertRaises(ValueError):
                check_plan(p)


if __name__ == '__main__':
    unittest.main()
