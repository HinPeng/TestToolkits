"""Required user inventory: public spelling, upstream JIT symbol, executable tests."""

NPU_TEST_FILES = ("test_operators.py", "test_extended_operators.py")

OPERATORS = {
    "split_qkv_rmsnorm_rope_kernel": ("Vllm", "split_qkv_rmsnorm_rope_kernel", "test_rope_eps_and_complete_versions"),
    "split_qkv_rmsnorm_mrope_kernel": ("Vllm", "split_qkv_rmsnorm_mrope_kernel", "test_split_qkv_mrope"),
    "split_qkv_rmsnorm_rope_simt_kernel": ("Vllm", "split_qkv_rmsnorm_rope_simt_kernel", "test_split_qkv_simt"),
    "_triton_rope": ("Vllm", "_triton_rope", "test_triton_rope_layouts"),
    "swiglu_quant_kernel": ("Vllm", "_swiglu_quant_kernel", "test_swiglu_quant_modes"),
    "swiglustep_kernel": ("Vllm", "_swiglustep_kernel", "test_swiglustep_limit"),
    "muls_add_kernel": ("Vllm", "muls_add_kernel", "test_muls_add_tile_reuse"),
    "apply_all_penalties_kernel": ("Vllm", "apply_all_penalties_kernel", "test_apply_all_penalties"),
    "token_bin_counts_and_mask_kernel": ("Vllm", "token_bin_counts_and_mask_kernel", "test_bincount_external_bound"),
    "build_local_metadata_triton": ("Vllm/MY", "build_local_metadata_triton", "test_metadata_capacity"),
    "rejection_random_sample_kernel": ("Vllm/MY", "rejection_random_sample_kernel", "test_rejection_modes_and_entropy"),
    "prepare_inputs_padded_kernel": ("Vllm/MY", "prepare_inputs_padded_kernel", "test_prepare_inputs_padded"),
    "rejection_greedy_sample_spec_len_1_triton": ("Vllm/MY", "rejection_greedy_sample_spec_len_1_triton", "test_greedy_spec_len_one"),
    "expand_kernel": ("Vllm/MY", "expand_kernel", "test_expand_signature_and_integer_width"),
    "triton_rms_kernel": ("Vllm/MY", "triton_rms_kernel", "test_rms_dim_static"),
    "rejection_greedy_sample_triton": ("Vllm/MY", "rejection_greedy_sample_triton", "test_greedy_variable_lengths"),
    "sample_recovered_tokens_kernel": ("Vllm/MY", "sample_recovered_tokens_kernel", "test_sample_recovered_tokens"),
    "_compute_slot_mapping_fused_groups_adaptive_kernel": ("Vllm/MY", "_compute_slot_mapping_fused_groups_adaptive_kernel", "test_slot_mapping_dynamic"),
    "_compute_slot_mapping_fused_groups_kernel": ("Vllm/MY", "_compute_slot_mapping_fused_groups_kernel", "test_slot_mapping_dynamic"),
}


def validate_operator_coverage(report):
    observed = {launch.get("kernel") for case in report["cases"].values() for launch in case["launches"]}
    missing = [name for name, (_, symbol, _) in OPERATORS.items() if symbol not in observed]
    if missing:
        raise ValueError("required operators were not executed: " + ", ".join(missing))
    return sorted(observed)
