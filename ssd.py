import numpy as np

# Ruiqi Yin - Final Project


def calculate_ssd(left_patch, right_patch):
    # For debug use:
    assert left_patch.shape == right_patch.shape

    return np.sum((left_patch - right_patch)**2)


def calculate_disparity_map(left_img, right_img, search_block_size, search_depth):
    # For debug use:
    assert left_img.shape == right_img.shape

    # print('left_img shape', left_img.shape)
    print('-------SSD Start-------')

    # parameter initialization
    disparity = []
    half_block_size = search_block_size // 2
    h, w, _ = left_img.shape

    for i in range(0, h):
        for j in range(0, w):
            # boundry checking, making sure when we shift, it would be inside
            if (i >= half_block_size and i < h - half_block_size) \
                    and (j >= half_block_size and j < w - half_block_size):
                min_ssd, min_disparity = float('inf'), 0
                for k in range(0, search_depth):
                    # Since right view, the search always going left
                    # if there are multiple same disparity, takes the minimum one
                    if j - k - half_block_size >= 0:
                        left_patch = left_img[i-half_block_size : i+half_block_size+1,
                                     j-half_block_size : j+half_block_size+1, :]
                        right_patch = right_img[i-half_block_size : i+half_block_size+1,
                                      j-k-half_block_size : j-k+half_block_size+1, :]
                        ssd_val = calculate_ssd(left_patch, right_patch)
                        # print(ssd_val)
                        if ssd_val < min_ssd:
                            min_ssd, min_disparity = ssd_val, k
                    else:
                        break
                disparity.append(min_disparity)

    disparity_map = np.array(disparity, dtype=np.float32)
    # reshape it to clip off the edges
    disparity_map = disparity_map.reshape(h-2*half_block_size, w-2*half_block_size)

    return disparity_map


