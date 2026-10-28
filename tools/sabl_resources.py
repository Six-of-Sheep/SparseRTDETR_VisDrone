"""Analytic Q300 capacity estimate; never imports a model/CUDA."""
import json,sys
from pathlib import Path

def estimate():
    b,q,c,h,w=8,300,256,7,7;n=b*q
    layers={'roi_FP32':n*c*h*w*4,'two_preconv_outputs_FP32':2*n*c*h*w*4,
            'two_axis_features_FP32':2*n*c*7*4,'two_up_features_FP32':2*n*c*14*4,
            'two_attention_maps_FP32':2*n*7*7*4,'two_prediction_tensors_FP32':2*n*4*7*4,
            'F8_original_BF16':b*c*96*168*2,'F8_FP32_cast':b*c*96*168*4,
            'roi_align_python_fallback_sample_tensor_FP32':n*c*7*7*2*2*4}
    mac={'two_preconv':2*n*7*7*c*c*9,'two_postconv':2*n*7*c*c*3,
         'two_transposeconv':2*n*7*c*c*2,'two_attention':2*n*7*7*c,
         'two_shared_linear_on_two_axes':2*2*n*14*c}
    total=sum(mac.values())
    return {'scope':'ANALYTIC_NOT_MEASURED','B':b,'Q':q,'RoI_count':n,'selected_tensor_bytes':layers,
            'selected_tensor_sum_bytes_NOT_peak':sum(layers.values()),'MAC_by_layer':mac,'head_forward_MAC':total,
            'head_forward_FLOPs_2_per_MAC':2*total,'per_image_head_MAC':total/b,
            'head_parameter_count':1837572,'S_A_parameter_count':21932156,
            'head_FP32_parameter_grad_AdamW_EMA_bytes':1837572*4*5,
            'S_A_FP32_parameter_grad_AdamW_EMA_bytes':21932156*4*5,
            'six_acceptance_checkpoints_state_lower_bound_bytes':2*(20094584+2*21932156)*4*4,
            'live_peak':'UNKNOWN: saved tensors, gradient liveness, convolution workspace, allocator, native/DN activations and deterministic RoIAlign backend require actual CUDA measurement',
            'time':'UNKNOWN: analytic MACs are not measured throughput or elapsed time',
            'storage':'No persistent RoI/features; keep 2 complete acceptance checkpoints per arm. 30GiB cap includes states, logs and temporary writes; estimate not a storage admission.'}
if __name__=='__main__':
    out=Path(sys.argv[1]);out.write_text(json.dumps(estimate(),indent=2)+'\n');print(json.dumps(estimate()))
