"""
augmentation.py  [Tripathy paper / standard MRI-CNN practice: rotation, flip, crop, contrast jitter]

Kept deliberately mild — large rotations/flips can be anatomically implausible for brain MRI
(e.g. horizontal flip mirrors left/right hemispheres, which is usually fine for OASIS axial
slices but double-check against your own slice orientation before trusting it blindly).
"""
import tensorflow as tf
from . import config as cfg

MAX_ROTATION_DEG = 10
CROP_FRACTION = 0.9
CONTRAST_RANGE = (0.85, 1.15)


def _random_rotate(img: tf.Tensor) -> tf.Tensor:
    # Small-angle rotation via tf.image (no extra deps like tensorflow-addons needed)
    angle_deg = tf.random.uniform([], -MAX_ROTATION_DEG, MAX_ROTATION_DEG)
    angle_rad = angle_deg * 3.14159265 / 180.0
    # Approximate small-angle rotation using shear-free affine via image transform
    cos_a = tf.cos(angle_rad)
    sin_a = tf.sin(angle_rad)
    h, w = cfg.IMG_SIZE, cfg.IMG_SIZE
    transform = [cos_a, -sin_a, (1 - cos_a) * w / 2 + sin_a * h / 2,
                 sin_a, cos_a, (1 - cos_a) * h / 2 - sin_a * w / 2,
                 0.0, 0.0]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=[transform],
        output_shape=[h, w],
        fill_value=0.0,
        interpolation="BILINEAR",
    )[0]
    return img


def _random_crop_resize(img: tf.Tensor) -> tf.Tensor:
    size = int(cfg.IMG_SIZE * CROP_FRACTION)
    img = tf.image.random_crop(img, size=[size, size, cfg.CHANNELS])
    img = tf.image.resize(img, [cfg.IMG_SIZE, cfg.IMG_SIZE])
    return img


def augment_image(img: tf.Tensor) -> tf.Tensor:
    img = tf.image.random_flip_left_right(img)
    img = _random_rotate(img)
    img = _random_crop_resize(img)
    img = tf.image.random_contrast(img, CONTRAST_RANGE[0], CONTRAST_RANGE[1])
    img.set_shape([cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.CHANNELS])
    return img


def ssl_augment_pair(img: tf.Tensor):
    """Two independently-augmented views of the same image, for contrastive SSL."""
    return augment_image(img), augment_image(img)
