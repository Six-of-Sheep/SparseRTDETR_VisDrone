"""R0 run identity; tested file/channel primitives reused unchanged."""
from ..localization.runio import (canonical,digest,sha,read,write,publish,atomic,event,reference,require,fsync_directory,original_process_exited)
TASK_ID="r0-seed0-e120-20261010-001"
PRETRAINED_SHA="911a745b62e173c8f4b9af513c2ea295428cf23f1bbfe9048381500f140fd720"
def execution_binding(c,mode):
    return {"task_id":TASK_ID,"arm":"R0","representation":"r0_scalar_sigmoid_go_v1","mode":mode,
            **{k:c[k] for k in ("contract_sha256","source_sha256","data_sha256","gpu_uuid","boot_id","environment","evaluation_protocol_sha256")}}
