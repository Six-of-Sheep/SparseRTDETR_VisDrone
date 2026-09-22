#!/usr/bin/env python3
"""Revision-031 identity wrapper for the unchanged frozen-plan verifier."""
from __future__ import annotations

import verify_v2b_frozen_launch_plan_r30 as _r30


_r30.AUTH_ID = "smoke-v2b-rev1-s2-r896-bridge-034"
_r30.PLAN_ID = "P3-V2B-REV1-FROZEN-LAUNCH-BRIDGE034"


if __name__ == "__main__":
    raise SystemExit(_r30.main())
