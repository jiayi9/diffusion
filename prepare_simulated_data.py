import os
import cv2
import numpy as np
import uuid
from typing import Tuple, Dict
from tqdm import tqdm
from matplotlib import pyplot as plt
import pylab


def simulate_random_shapes(
    shape: str,
    output_folder: str,
    sample_size: int,
    length_mean: float = -1,
    length_std: float = -1,
    thickness: int = -1,
    radius_mean: float = -1,
    radius_std: float = -1,
    scale: Tuple[int, int] = (512, 512),
    noise_ratio: float = 0.01,
    border_ratio: Tuple[float, float] = (0.2, 0.8),
    image_format: str = "png",
    convert_to_rgb: bool = False,
    generate_mask: bool = False,
    mask_format: str = "npz",
    one_hot_mask_encode: Dict[str, int] = None,
) -> None:
    """
    This function generate random images of basic geometric shapes with a noisy background.
    The shapes include line segments and circles. The purpose is to generate sample data for basic tests.

    Args:
        shape: the geometric shapes to generate
        output_folder: the folder to save random images
        sample_size: the number of random images to generate
        length_mean: mean length for shape line
        length_std: std of length for shape line
        thickness: thickness for shape line
        radius_mean: mean radius for shape circle
        radius_std: std pf mean radius for shape circle
        scale: the height and width of the image to generate
        noise_ratio: the proportion of noisy points
        border_ratio: where the starting, ending points or circle center are not in
        image_format: the image format to save in
        convert_to_rgb: convert grayscale to rgb channel
        generate_mask: bool indicator for mask generation
        mask_format: the image format to save in (npz recommended)
        one_hot_mask_encode: the information to map classes to the mask layers

    Returns:
        None

    """
    # create the saving folder if not existent
    os.makedirs(os.path.join(output_folder), exist_ok=True)
    # generate images through loop
    for i in tqdm(range(sample_size)):
        img = np.random.binomial(1, noise_ratio, scale).astype(float)
        if generate_mask:
            if one_hot_mask_encode:
                num_classes = len(one_hot_mask_encode)
                scale_ext = list(scale)
                scale_ext.insert(0, num_classes)
                mask = np.zeros(scale_ext, np.uint8)
            else:
                mask = np.zeros(scale, np.uint8)
        if shape == "line":
            # Need randomize the starting and ending coordinates for line
            starting_point = np.append(
                np.random.uniform(
                    scale[0] * border_ratio[0], scale[0] * border_ratio[1], 1
                ).astype(int),
                np.random.uniform(
                    scale[0] * border_ratio[0], scale[1] * border_ratio[1], 1
                ).astype(int),
            )
            if length_mean < 0:
                length_mean = scale[0] / 10
            if length_std < 0:
                length_std = scale[0] / 100
            ending_point = tuple(
                starting_point
                + length_mean * np.random.normal(0, length_std, 2).astype(int)
            )
            img_new = cv2.line(
                img, starting_point, ending_point, color=1, thickness=thickness
            )
            if generate_mask:
                if one_hot_mask_encode:
                    class_index = one_hot_mask_encode[shape]
                    mask[class_index, :, :] = cv2.line(
                        mask[class_index, :, :],
                        starting_point,
                        ending_point,
                        color=1,
                        thickness=thickness + 1,
                    )
                else:
                    mask = cv2.line(
                        mask,
                        starting_point,
                        ending_point,
                        color=1,
                        thickness=thickness + 1,
                    )
                mask_new = mask
        elif shape == "circle":
            # Need to randomize the center coordinate and the radius value
            center = np.append(
                np.random.uniform(
                    scale[0] * border_ratio[0], scale[0] * border_ratio[1], 1
                ).astype(int),
                np.random.uniform(
                    scale[0] * border_ratio[0], scale[1] * border_ratio[1], 1
                ).astype(int),
            )
            center = tuple(center)
            radius = max(int((np.random.normal(0, radius_std, 1) + radius_mean)[0]), 2)
            img_new = cv2.circle(img, center, radius, 1, -1)
            if generate_mask:
                if one_hot_mask_encode:
                    class_index = one_hot_mask_encode[shape]
                    mask[class_index, :, :] = cv2.circle(
                        mask[class_index, :, :], center, radius + 1, 1, -1
                    )
                else:
                    mask = cv2.circle(mask, center, radius + 1, 1, -1)
                mask_new = mask
        elif shape == "square":
            # Recommend keeping this structure for future work
            print(
                "The shape specified is wrong or not included yet. No images are generated."
            )
            return
        elif shape == "triangle":
            # Recommend keeping this structure for future work
            print(
                "The shape specified is wrong or not included yet. No images are generated."
            )
            return
        elif shape == "noise":
            img_new = img
            if generate_mask:
                mask_new = mask
        else:
            print(
                "The shape specified is wrong or not included yet. No images are generated."
            )
            return
        if convert_to_rgb:
            # Convert (height, width) to (height, width, 3) with duplications
            img_new = np.stack((img_new, img_new, img_new), axis=2)

        uuid_str = uuid.uuid4().hex

        # save image
        img_output_path = os.path.join(output_folder, uuid_str + "." + image_format)

        cv2.imwrite(img_output_path, img_new * 255)

        # multiplier = np.random.randint(128, high=255, size=img_new.shape, dtype=int)
        # product = img_new * multiplier
        # cv2.imwrite(img_output_path, product)


        # save mask
        mask_output_path = os.path.join(
            output_folder, uuid_str + "_mask." + mask_format
        )
        if generate_mask:
            if mask_format == "npz":
                if one_hot_mask_encode:
                    np.savez_compressed(mask_output_path, mask=mask_new)
                else:
                    mask_new = np.expand_dims(mask_new, 0)
                    np.savez_compressed(mask_output_path, mask=mask_new)
            else:
                cv2.imwrite(mask_output_path, mask_new * 255)


def quick_simulate(
    data_folder: str,
    sample_size: int,
    image_size: Tuple[int, int],
    image_format: str = "png",
    convert_to_rgb: bool = True,
    generate_mask: bool = False,
    mask_format: str = "npz",
    one_hot_mask_encode: Dict[str, int] = None,
) -> None:
    """
    Quick simulation of random images of three classes (lines, circles and pure noises)
    Args:
        data_folder: the data folder where the simulated random images are saved
        sample_size: the sample size for each class or shape
        image_size: the height and width of the image, can be int or (int, int)
        image_format: format of the output image
        convert_to_rgb: rgb or grayscale
        generate_mask: whether a mask is generated for segmentation models
        mask_format: recommend using npz
        one_hot_mask_encode: the information to map classes to the mask layers

    Returns:
        None

    """
    scale = (image_size, image_size) if isinstance(image_size, int) else image_size
    simulate_random_shapes(
        output_folder=os.path.join(data_folder, "circle"),
        sample_size=sample_size,
        shape="circle",
        radius_mean=25,
        radius_std=2,
        scale=scale,
        convert_to_rgb=convert_to_rgb,
        image_format=image_format,
        generate_mask=generate_mask,
        mask_format=mask_format,
        one_hot_mask_encode=one_hot_mask_encode,
    )
    simulate_random_shapes(
        output_folder=os.path.join(data_folder, "line"),
        sample_size=sample_size,
        shape="line",
        length_mean=55,
        length_std=10,
        thickness=2,
        scale=scale,
        convert_to_rgb=convert_to_rgb,
        image_format=image_format,
        generate_mask=generate_mask,
        mask_format=mask_format,
        one_hot_mask_encode=one_hot_mask_encode,
    )
    simulate_random_shapes(
        output_folder=os.path.join(data_folder, "noise"),
        sample_size=sample_size,
        shape="noise",
        scale=scale,
        convert_to_rgb=convert_to_rgb,
        image_format=image_format,
        generate_mask=generate_mask,
        mask_format=mask_format,
        one_hot_mask_encode=one_hot_mask_encode,
    )


def visualize_npz_mask(
    file_path: str, concat_axis: int = 1, padding_thickness: int = 1, plot: bool = True
) -> np.array:
    """visualize the one-hot encoding (layers or masks of different classes) of a npz mask file"""
    mask_npy = np.load(file_path)["mask"]
    _, height, width = mask_npy.shape
    padding_size = (
        (padding_thickness, width) if concat_axis == 0 else (height, padding_thickness)
    )
    padding = np.ones(padding_size)
    mask_concat = np.concatenate(
        [np.concatenate([layer, padding], axis=concat_axis) for layer in mask_npy],
        axis=concat_axis,
    )
    if plot:
        plt.imshow(mask_concat)
        pylab.show()
    return mask_concat


quick_simulate('D:/Temp/simulated_shapes', 10000, (128, 128), convert_to_rgb=True)


#######################################################################################################################


#
#
# data_path = "C:/temp/simulated_shapes"
# dataset = create_classification_df(data_path)
# clsdataset = ClsDataset(dataset)
#
# clsdataloader = DataLoader(
#     clsdataset,
#     batch_size=2,
#     shuffle=True,
#     num_workers=0,
#     pin_memory=True,
# )
