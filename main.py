import cv2
import numpy as np
import ssd as ssd
import os
import graphcut
from skimage.transform import resize


# Ruiqi Yin - Final Project

# GLOBAL Variables
INPUT_DIR = 'input'
OUTPUT_DIR = 'output'

# image name, resize, search_block_size, search_depth
IMAGES = {
    'Backpack': (0.2, 10, 50),
    'Piano': (0.2, 10, 50),
    'Umbrella': (0.1, 10, 100),
    'Bicycle': (0.2, 10, 50),
    # 'Flowers': (0.5, 10, 70),
    # 'Umbrella': (0.04, 10, 150),

}

def load_image(name, flag, resize=1):
    img = cv2.imread(name, flag)
    img = cv2.resize(img, None, fx=resize, fy=resize)
    return img


def convert_disparity_to_grayscale(disparity):
    # Identify  where disparity values are less than 0
    negatives = disparity < 0
    size = (disparity.shape[0], disparity.shape[1], 3)
    image = np.zeros(size, dtype=np.uint8)
    # Scale and clip the disparity values to the range [0, 255]
    upper = 255 * disparity / disparity.max()
    upper = 255 * disparity / 43
    image[:] = np.clip(upper, 0, 255)[:, :, np.newaxis]
    # Assign the scaled disparity values to the image channels
    image[:] = np.where(negatives, 0, upper)[:, :, np.newaxis]
    # Set the pixels with negative disparity values to a specific color
    image[negatives] = [255, 255, 0]

    return image


def ssd_get_disparity(img_name):
    # load image and its needed meta data
    resize, search_block_size, search_depth = IMAGES[img_name]
    print('load left image')
    left = load_image(os.path.join(INPUT_DIR, img_name, 'im0.png'), cv2.IMREAD_COLOR, resize=resize)
    print('load right image')
    right = load_image(os.path.join(INPUT_DIR, img_name, 'im1.png'),  cv2.IMREAD_COLOR, resize=resize)
    disparity_map = ssd.calculate_disparity_map(left, right, search_block_size, search_depth)

    return disparity_map

def graph_get_disparity(img_name):
    # load image and its needed meta data
    resize, search_block_size, search_depth = IMAGES[img_name]
    # TODO: Load with color or grayscale
    flag = cv2.IMREAD_COLOR
    left = load_image(os.path.join(INPUT_DIR, img_name, 'im0.png'),  flag=flag, resize=resize)
    print('load right image')
    right = load_image(os.path.join(INPUT_DIR, img_name, 'im1.png'),  flag=flag, resize=resize)
    print('load ground truth')
    ground_truth = cv2.imread(os.path.join(INPUT_DIR, img_name, 'disp0.png'), cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.resize(ground_truth, None, fx=resize, fy=resize)

    # # TODO: another param to tune
    ground_truth = np.round(ground_truth / 6).astype(np.uint8)
    max_disparity = ground_truth.max() + 1
    #
    # TODO: kernel size, dispairty step to tune
    graph_cut = graphcut.GraphCut(left, right, max_disparity, kernel=3, disparity_step=1)
    disparity_map = graph_cut.perform_alpha_expansion()


    return disparity_map

# Measure Performance
def pixelwise_difference(disparity, groundtruth):
    common_shape = (min(disparity.shape[0], groundtruth.shape[0]),
                    min(disparity.shape[1], groundtruth.shape[1]))

    disparity_resized = resize(disparity, common_shape, mode='reflect', anti_aliasing=True)
    groundtruth_resized = resize(groundtruth, common_shape, mode='reflect', anti_aliasing=True)

    diff = np.abs(disparity_resized - groundtruth_resized)
    avg_diff = np.mean(diff)

    return avg_diff

def get_correlation(disparity, groundtruth):
    # Reshape the images in case the shape are not the same
    common_shape = (min(disparity.shape[0], groundtruth.shape[0]),
                    min(disparity.shape[1], groundtruth.shape[1]))
    disparity_resized = resize(disparity, common_shape, mode='reflect', anti_aliasing=True)
    groundtruth_resized = resize(groundtruth, common_shape, mode='reflect', anti_aliasing=True)

    disparity_resized = disparity_resized.astype(np.float32)
    groundtruth_resized = groundtruth_resized.astype(np.float32)

    (dis_mean, dis_std) = cv2.meanStdDev(disparity_resized)
    (gt_mean, gt_std) = cv2.meanStdDev(groundtruth_resized)

    # get the normalized cross-correlation
    num_pixels = common_shape[0] * common_shape[1]
    elementwise_product = np.multiply(disparity_resized, groundtruth_resized)
    num = np.sum(elementwise_product) - num_pixels * dis_mean[0] * gt_mean[0]

    denom = (num_pixels - 1) * gt_std[0] * dis_std[0]
    score = num / denom

    return score



if __name__ == '__main__':
    image_name = 'Umbrella'
    method = 'ssd'
    if method == 'ssd':
        disparity_map = ssd_get_disparity(image_name)
    else:
        disparity_map = graph_get_disparity(image_name)

    disparity_gray = convert_disparity_to_grayscale(disparity_map)
    output_filename = os.path.join(OUTPUT_DIR, f"{image_name}-{method}-disparity-grayscale.png")
    cv2.imwrite(output_filename, disparity_gray)

    colormap_image = cv2.applyColorMap(disparity_gray, cv2.COLORMAP_JET)
    output_filename = os.path.join(OUTPUT_DIR, f"{image_name}-{method}-disparity-jet.png")
    cv2.imwrite(output_filename, colormap_image)

    # compare metrics
    if False:
        output_filename_ssd = os.path.join(OUTPUT_DIR, f"{image_name}-ssd-disparity-grayscale.png")
        output_filename_graph = os.path.join(OUTPUT_DIR, f"{image_name}-graph-disparity-grayscale.png")

        flag = cv2.IMREAD_GRAYSCALE
        disparity_ssd = load_image(output_filename_ssd, flag, resize=1)
        disparity_graph = load_image(output_filename_graph, flag, resize=1)
        ground_truth = cv2.imread(os.path.join(INPUT_DIR, image_name, 'disp0.png'), cv2.IMREAD_GRAYSCALE)
        print(disparity_ssd.shape)
        print(disparity_graph.shape)
        diff1 = pixelwise_difference(disparity_ssd, ground_truth)
        diff2 = pixelwise_difference(disparity_graph, ground_truth)
        diff3 = pixelwise_difference(disparity_graph, disparity_ssd)

        print(diff1, diff2, diff3)

        print(get_correlation(disparity_ssd, ground_truth))
        print(get_correlation(disparity_graph, ground_truth))
