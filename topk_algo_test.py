def topk(seq, K_pixel_points):
    collected_id = [None]
    j = 0
    Kpps = K_pixel_points
    pps_alpha = [None] * Kpps
    pps_id = [None] * Kpps
    kpps = 0
    # Use a min-heap to keep track of the top-K alpha values
    for collected_id[j], alpha in enumerate(seq):
        if kpps < Kpps:
            # If the heap is not full, add the current point
            cur_idx = kpps
            pps_alpha[cur_idx] = alpha
            pps_id[cur_idx] = collected_id[j]
            for k in range(K_pixel_points):
                if cur_idx == 0:
                    break
                parent = (cur_idx - 1) >> 1
                if pps_alpha[parent] > pps_alpha[cur_idx]:
                    # If the parent is smaller, swap current with parent
                    tmp_alpha = pps_alpha[cur_idx]
                    tmp_id = pps_id[cur_idx]
                    pps_alpha[cur_idx] = pps_alpha[parent]
                    pps_id[cur_idx] = pps_id[parent]
                    pps_alpha[parent] = tmp_alpha
                    pps_id[parent] = tmp_id
                    cur_idx = parent
                else:
                    break
            kpps += 1
        elif alpha > pps_alpha[0]:
            cur_idx = 0
            next = 0
            pps_alpha[0] = alpha
            pps_id[0] = collected_id[j]
            for k in range(K_pixel_points):
                l_idx = (cur_idx << 1) + 1
                r_idx = (cur_idx << 1) + 2
                if l_idx >= Kpps:
                    break
                elif r_idx >= Kpps or pps_alpha[l_idx] < pps_alpha[r_idx]:
                    next = l_idx
                else:
                    next = r_idx
                if pps_alpha[next] < pps_alpha[cur_idx]:
                    # If the child is larger, swap current with child
                    tmp_alpha = pps_alpha[cur_idx]
                    tmp_id = pps_id[cur_idx]
                    pps_alpha[cur_idx] = pps_alpha[next]
                    pps_id[cur_idx] = pps_id[next]
                    pps_alpha[next] = tmp_alpha
                    pps_id[next] = tmp_id
                    cur_idx = next
                else:
                    break
    return pps_alpha, pps_id
import numpy as np

seq = np.random.rand(1000)
K_pixel_points = 8
print(topk(seq, K_pixel_points))
print(np.argsort(seq)[::-1][:K_pixel_points])
print(np.sort(seq)[::-1][:K_pixel_points])