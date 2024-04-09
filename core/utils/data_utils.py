import cv2
import numpy as np
from scipy.spatial.distance import cdist


def read_image_cv2(file_name, format=None):
    """# NOTE modified from detectron2, use cv2 instead of PIL Read an image
    into the given format.

    Args:
        file_name (str): image file path
        format (str): "BGR" | "RGB"
    Returns:
        image (np.ndarray): an HWC image
    """
    image = cv2.imread(file_name, cv2.IMREAD_COLOR)  # BGR
    if format == "RGB":
        # flip channels if needed
        image = image[:, :, [2, 1, 0]]
    if format == "L":
        image = np.expand_dims(image, -1)  # TODO: debug this when needed
    return image


def denormalize_image(image, cfg):
    # CHW
    pixel_mean = np.array(cfg.MODEL.PIXEL_MEAN).reshape(-1, 1, 1)
    pixel_std = np.array(cfg.MODEL.PIXEL_STD).reshape(-1, 1, 1)
    return image * pixel_std + pixel_mean


def crop_resize_by_d2_roialign(
    img,
    center,
    scale,
    output_size,
    aligned=True,
    interpolation="bilinear",
    in_format="HWC",
    out_format="HWC",
    dtype="float32",
):
    """
    img: HWC
    output_size: int or (w, h)
    """
    import torch
    from detectron2.layers.roi_align import ROIAlign
    from torchvision.ops import RoIPool

    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    # NOTE: different to cv2 convention!!!
    output_size = (output_size[1], output_size[0])  # to (h, w)
    if interpolation == "bilinear":
        op = ROIAlign(output_size, 1.0, 0, aligned=aligned)
    elif interpolation == "nearest":
        op = RoIPool(output_size, 1.0)  #
    else:
        raise ValueError(f"Wrong interpolation type: {interpolation}")

    assert in_format in ["HW", "HWC", "CHW"]
    if in_format == "HW":
        img = img[None]
    elif in_format == "HWC":
        img = img.transpose(2, 0, 1)  # CHW

    img_tensor = torch.tensor(img[None].astype("float32"))
    cx, cy = center
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    bw, bh = scale
    rois = torch.tensor(np.array([0] + [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dtype="float32")[None])
    result = op(img_tensor, rois)[0].numpy().astype(dtype)
    if out_format == "HWC":
        result = result.transpose(1, 2, 0)
    return result

"""
his function performs the cropping and resizing of an image based on the specified center, scale, output size, and rotation angle. It also allows for specifying the interpolation method used during the resizing process.

Parameters:
img: The input image to transform.
center: The center point (cx, cy) around which the image will be cropped and/or rotated.
scale: A tuple (w, h) or a single value defining the scaling factor. If a single value is provided, it's assumed to be the same for both width and height.
output_size: The size (w, h) of the output image. If a single integer is given, it is used for both dimensions, resulting in a square output.
rot: The rotation angle in degrees. Default is 0, meaning no rotation.
interpolation: The interpolation method used in the resizing process. Default is cv2.INTER_LINEAR.
Process:
It adjusts the scale and output_size parameters to ensure they are in tuple form (w, h).
Calls get_affine_transform to generate the affine transformation matrix based on the specified parameters.
Uses cv2.warpAffine with the transformation matrix to perform the actual image transformation, resulting in the cropped, resized, and optionally rotated image.
"""

def crop_resize_by_warp_affine(img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR, bbox=None, rnd_bg=None):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    trans = get_affine_transform(center, scale, rot, output_size)

    # Apply affine transformation to the image and the mask
    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)

    if bbox is not None:
        #print(bbox)
        bbox = np.round(bbox).astype(int)
        mask = np.zeros(img.shape[:2], dtype=np.uint8) 
        mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255
        dst_mask = cv2.warpAffine(mask, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)
        if rnd_bg is not None:
            for i in range(dst_img.shape[0]):
                for j in range(dst_img.shape[1]):
                    if dst_mask[i,j] == 0:
                        dst_img[i,j] = rnd_bg[i,j]
        else:
            dst_img[dst_mask==0] = 0

    return dst_img

"""
This function calculates the affine transformation matrix required to perform the desired transformations on the image.

Parameters:
center, scale, rot, output_size: Same as in crop_resize_by_warp_affine.
shift: An optional shift to apply before scaling, useful for fine adjustments. Defaults to [0, 0].
inv: If True, calculates the inverse affine transformation. This could be useful for reversing the transformation.
Process:
Converts input parameters to the appropriate data types and formats.
Calculates the rotation in radians and determines the direction vectors before and after rotation.
Sets up source (src) and destination (dst) points for the affine transformation. These are based on the center, scale, and rotation parameters, as well as the output size.
If inv is True, calculates the affine transform matrix for the inverse transformation, otherwise calculates the normal affine transform matrix.
This matrix is then used to map the source points to the destination points, effectively performing the desired transformation when applied to an image.
"""

def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=False):
    """
    adapted from CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    center: ndarray: (cx, cy)
    scale: (w, h)
    rot: angle in deg
    output_size: int or (w, h)
    """
    if isinstance(center, (tuple, list)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)

    if isinstance(output_size, (int, float)):
        output_size = (output_size, output_size)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def resize_short_edge(im, target_size, max_size, stride=0, interpolation=cv2.INTER_LINEAR, return_scale=False):
    """Scale the shorter edge to the given size, with a limit of `max_size` on
    the longer edge. If `max_size` is reached, then downscale so that the
    longer edge does not exceed max_size. only resize input image to target
    size and return scale.

    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        if return_scale:
            return im, im_scale
        else:
            return im
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[: im.shape[0], : im.shape[1], :] = im
        if return_scale:
            return padded_im, im_scale
        else:
            return padded_im


def get_fps_and_center(pts, num_fps=8, init_center=True):
    """get fps points + center."""
    from core.csrc.fps.fps_utils import farthest_point_sampling

    avgx = np.average(pts[:, 0])
    avgy = np.average(pts[:, 1])
    avgz = np.average(pts[:, 2])
    fps_pts = farthest_point_sampling(pts, num_fps, init_center=init_center)
    res_pts = np.concatenate([fps_pts, np.array([[avgx, avgy, avgz]])], axis=0)
    return res_pts


def xyz_to_region(xyz_crop, fps_points):
    bh, bw = xyz_crop.shape[:2]
    mask_crop = ((xyz_crop[:, :, 0] != 0) | (xyz_crop[:, :, 1] != 0) | (xyz_crop[:, :, 2] != 0)).astype("uint8")
    dists = cdist(xyz_crop.reshape(bh * bw, 3), fps_points)  # (hw, f)
    region_ids = np.argmin(dists, axis=1).reshape(bh, bw) + 1  # NOTE: 1 to num_fps
    # (bh, bw)
    return mask_crop * region_ids  # 0 means bg


def get_2d_coord_np(width, height, low=0, high=1, fmt="CHW"):
    """
    Args:
        width:
        height:
    Returns:
        xy: (2, height, width)
    """
    # coords values are in [low, high]  [0,1] or [-1,1]
    x = np.linspace(low, high, width, dtype=np.float32)
    y = np.linspace(low, high, height, dtype=np.float32)
    xy = np.asarray(np.meshgrid(x, y))
    if fmt == "HWC":
        xy = xy.transpose(1, 2, 0)
    elif fmt == "CHW":
        pass
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return xy


# tests --------------------------------------------------------------------------
def test_get_2d_coord():
    xy = get_2d_coord_np(width=640, height=480)
    print(xy.shape, xy.dtype)


if __name__ == "__main__":
    test_get_2d_coord()
