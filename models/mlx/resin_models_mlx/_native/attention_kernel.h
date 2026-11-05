// Shared verbatim partition body for native and generic MLX resource binding.
#if defined(RESIN_ATTENTION_BODY)

    constexpr int LANE_DIMS = D / 32;
    const uint lane = thread_index_in_simdgroup;
    const uint group = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    // Runtime geometry: page-table width, partition count, layer offset and
    // the first readable history position (earlier slots are never read).
    const int PAGES = geometry[0];
    const int PARTITIONS = geometry[1];
    const int layer = geometry[2];
    const int skip = geometry[3];

    const int work = threadgroup_position_in_grid.z;
    const int partition = work % PARTITIONS;
    const int head_group = (work / PARTITIONS) % HEAD_GROUPS;
    const int kv_head = (work / (PARTITIONS * HEAD_GROUPS)) % KV_HEADS;
    const int row = work / (PARTITIONS * HEAD_GROUPS * KV_HEADS);
    const int first_head = kv_head * (HEADS / KV_HEADS) + head_group * GROUP;

    const int history = lengths[row];
    const int total = history + 1;
    const int start = partition * PART;
    const int stop = min(start + PART, total);

    float q[GROUP][LANE_DIMS];
    for (int h = 0; h < GROUP; ++h) {
        for (int i = 0; i < LANE_DIMS; ++i) {
            q[h][i] = float(queries[((row * HEADS) + first_head + h) * D
                                    + lane * LANE_DIMS + i]) * scale[0];
        }
    }
    float m[GROUP];
    float l[GROUP];
    float acc[GROUP][LANE_DIMS];
    for (int h = 0; h < GROUP; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
        for (int i = 0; i < LANE_DIMS; ++i) acc[h][i] = 0.0f;
    }

    // Each simdgroup walks its own contiguous run of positions, page by page,
    // loading PREFETCH positions ahead so their memory latency overlaps.
    const int width = KV_HEADS * D;
    const int run = PART / SIMDGROUPS;
    const int first = start + int(group) * run;
    const int last = min(first + run, stop);
    const int cached = min(last, history);
    int p = max(first, skip);
    while (p < cached) {
        const int page_end = min(cached, (p / PAGE + 1) * PAGE);
        const long unit = long(pages[row * PAGES + p / PAGE]) + layer;
        const device T* key = units + (unit * 2 * PAGE + p % PAGE) * width
                              + kv_head * D + lane * LANE_DIMS;
        const device T* value = key + long(PAGE) * width;
        for (; p + PREFETCH <= page_end; p += PREFETCH) {
            float k[PREFETCH][LANE_DIMS];
            float v[PREFETCH][LANE_DIMS];
            for (int t = 0; t < PREFETCH; ++t) {
                for (int i = 0; i < LANE_DIMS; ++i) k[t][i] = float(key[t * width + i]);
            }
            for (int t = 0; t < PREFETCH; ++t) {
                for (int i = 0; i < LANE_DIMS; ++i) v[t][i] = float(value[t * width + i]);
            }
            for (int t = 0; t < PREFETCH; ++t) attend(q, k[t], v[t], m, l, acc);
            key += PREFETCH * width;
            value += PREFETCH * width;
        }
        for (; p < page_end; ++p) {
            float k[LANE_DIMS];
            float v[LANE_DIMS];
            for (int i = 0; i < LANE_DIMS; ++i) {
                k[i] = float(key[i]);
                v[i] = float(value[i]);
            }
            attend(q, k, v, m, l, acc);
            key += width;
            value += width;
        }
    }
    // The step's own key and value follow the history.
    if (first <= history && history < last) {
        const int base = (row * KV_HEADS + kv_head) * D + lane * LANE_DIMS;
        float k[LANE_DIMS];
        float v[LANE_DIMS];
        for (int i = 0; i < LANE_DIMS; ++i) {
            k[i] = float(current_keys[base + i]);
            v[i] = float(current_values[base + i]);
        }
        attend(q, k, v, m, l, acc);
    }

    threadgroup float shared_m[SIMDGROUPS][GROUP];
    threadgroup float shared_l[SIMDGROUPS][GROUP];
    threadgroup float shared_acc[SIMDGROUPS][GROUP][D];
    for (int h = 0; h < GROUP; ++h) {
        if (lane == 0) {
            shared_m[group][h] = m[h];
            shared_l[group][h] = l[h];
        }
        for (int i = 0; i < LANE_DIMS; ++i) {
            shared_acc[group][h][lane * LANE_DIMS + i] = acc[h][i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int index = tid; index < GROUP * D; index += SIMDGROUPS * 32) {
        const int h = index / D;
        const int d = index % D;
        float peak = -INFINITY;
        for (int s = 0; s < SIMDGROUPS; ++s) peak = max(peak, shared_m[s][h]);
        float sum = 0.0f;
        float value = 0.0f;
        if (peak != -INFINITY) {
            for (int s = 0; s < SIMDGROUPS; ++s) {
                const float scaled = fast::exp(shared_m[s][h] - peak);
                sum += shared_l[s][h] * scaled;
                value += shared_acc[s][h][d] * scaled;
            }
        }
        const int out = ((row * HEADS) + first_head + h) * PARTITIONS + partition;
        partial[out * D + d] = value;
        if (d == 0) {
            maxima[out] = peak;
            sums[out] = sum;
        }
    }

#else

template <int GROUP, int LANE_DIMS>
inline void attend(
    thread const float (&q)[GROUP][LANE_DIMS],
    thread const float (&k)[LANE_DIMS],
    thread const float (&v)[LANE_DIMS],
    thread float (&m)[GROUP],
    thread float (&l)[GROUP],
    thread float (&acc)[GROUP][LANE_DIMS]) {
    for (int h = 0; h < GROUP; ++h) {
        float dot = 0.0f;
        for (int i = 0; i < LANE_DIMS; ++i) dot += q[h][i] * k[i];
        const float score = simd_sum(dot);
        const float peak = max(m[h], score);
        const float correction = fast::exp(m[h] - peak);
        const float weight = fast::exp(score - peak);
        l[h] = l[h] * correction + weight;
        for (int i = 0; i < LANE_DIMS; ++i) {
            acc[h][i] = acc[h][i] * correction + weight * v[i];
        }
        m[h] = peak;
    }
}

#endif
