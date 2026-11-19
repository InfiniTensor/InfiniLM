"""Freeze the supported fresh B1 VideoNSA media input before native submission."""
MEDIA = ('pixel_values', 'image_bound', 'tgt_sizes', 'image_req_ids', 'visual_token_ranges')


def freeze_video_input(engine, snapshot, data, core):
    if not any(data.get(k) is not None for k in (*MEDIA, 'image_grid_thw')):
        return
    if data.get('image_grid_thw') is not None or any(data.get(k) is None for k in MEDIA):
        raise ValueError('NSA video requires complete processor media metadata')
    if len(snapshot.rows) != 1:
        raise ValueError('NSA video supports one fresh request')
    row = snapshot.rows[0]
    if row.past != 0 or not 2 <= row.total <= 8192:
        raise ValueError('NSA video requires complete uncached prefill of 2..8192 tokens')
    if data['image_req_ids'] != [0] or any(len(data[k]) != 1 for k in MEDIA[:3]):
        raise ValueError('NSA video media must belong to the single request')
    config = engine.hf_config
    vision = config['vision_config']
    merge = int(vision['spatial_merge_size'])
    patch = int(vision['in_channels']) * int(vision['temporal_patch_size']) * int(vision['patch_size']) ** 2
    if merge != 2 or patch != 1176:
        raise ValueError('NSA video geometry outside the verified model')
    pixels, bounds, grids = (data[k][0] for k in MEDIA[:3])
    def check(t, dtype, rank, name):
        if t is None or t.device.type != 'cpu' or t.dtype != dtype or t.ndim != rank or not t.is_contiguous():
            raise ValueError('NSA video requires contiguous CPU ' + name)
    dtype_name = config.get('torch_dtype') or config.get('dtype')
    dtype = {'bfloat16': core.bfloat16, 'float16': core.float16}.get(dtype_name)
    if dtype is None:
        raise ValueError('NSA video requires FP16 or BF16 model dtype')
    check(pixels, dtype, 2, 'pixels')
    check(bounds, core.int64, 3, 'bounds')
    check(grids, core.int64, 2, 'grids')
    if bounds.shape[0] != 1 or bounds.shape[2] != 2 or grids.shape[1] != 3 or bounds.shape[1] != grids.shape[0] or grids.shape[0] == 0:
        raise ValueError('NSA video bounds/grid shapes differ')
    bv, gv = bounds.to_numpy().tolist()[0], grids.to_numpy().tolist()
    patches, previous, ranges = 0, 0, []
    for (start, end), (t, h, w) in zip(bv, gv):
        if min(t, h, w) <= 0 or h % merge or w % merge or not previous <= start < end <= row.total:
            raise ValueError('NSA video grid/bounds invalid')
        count = t * h * w
        if end - start != count // (merge * merge):
            raise ValueError('NSA video replacement length differs from grid')
        patches += count
        previous = end
        ranges.extend((start, end))
    if tuple(pixels.shape) != (patches, patch) or patches > 4 * row.total or list(data['visual_token_ranges']) != ranges:
        raise ValueError('NSA video pixels/ranges differ from replacement bounds')
    positions = data.get('position_ids')
    check(positions, core.int64, 2, 'positions')
    limit = config.get('max_position_embeddings', config.get('text_config', {}).get('max_position_embeddings'))
    pv = positions.to_numpy()
    if tuple(positions.shape) != (3, row.total) or pv.min() < 0 or pv.max() >= limit:
        raise ValueError('NSA video position ids outside rotary cache')
    # CPU copy_ supports BF16 without numpy conversion and severs writable aliases.
    for key, tensor in zip(MEDIA[:3], (pixels, bounds, grids)):
        private = core.empty(tensor.shape, dtype=tensor.dtype)
        private.copy_(tensor)
        data[key] = [private]
    data['image_req_ids'] = [0]
    data['visual_token_ranges'] = ranges
