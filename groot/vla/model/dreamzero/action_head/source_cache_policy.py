def _should_reset_for_local_attention(
    *,
    current_start_frame: int,
    local_attn_size: int,
    persistent_source_cache: bool,
) -> bool:
    if persistent_source_cache:
        return False
    if local_attn_size <= 0:
        return False
    return current_start_frame >= local_attn_size
