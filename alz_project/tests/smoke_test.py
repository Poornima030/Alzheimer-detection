"""
smoke_test.py — sanity-checks every model builds and runs on random dummy tensors of the
correct shape. This does NOT verify accuracy/learning (no real MRI data here) — it only
verifies the code runs end-to-end without shape/API errors, which is what can actually be
checked in an environment with no GPU, no dataset, and no internet access to Kaggle.

Run:
    python -m tests.smoke_test
"""
import numpy as np
import tensorflow as tf

from src import config as cfg
from src.models.fusion_model import build_model, VALID_EXPERIMENTS
from src.models import ssl_pretrain as ssl
from src.explainability.gradcam import make_gradcam_heatmap, find_last_conv_layer_name


def dummy_batch(batch_size=2):
    return np.random.rand(batch_size, cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.CHANNELS).astype("float32")


def test_all_experiment_architectures():
    for exp in VALID_EXPERIMENTS:
        build_name = "cbam_transformer" if exp == "cbam_transformer_ssl" else exp
        model = build_model(build_name)
        out = model(dummy_batch())
        assert out.shape == (2, cfg.NUM_CLASSES), f"{exp}: bad output shape {out.shape}"
        assert np.allclose(np.sum(out.numpy(), axis=1), 1.0, atol=1e-3), f"{exp}: softmax doesn't sum to 1"
        print(f"[OK] {exp}: builds, forward pass shape {out.shape}, params={model.count_params():,}")


def test_ssl_forward_pass():
    ssl_model, backbone_model = ssl.build_ssl_model()
    z = ssl_model(dummy_batch(4))
    assert z.shape == (4, cfg.SSL_PROJECTION_DIM)
    loss = ssl.nt_xent_loss(z[:2], z[2:])
    assert loss.shape == ()
    print(f"[OK] SSL model: projection shape {z.shape}, NT-Xent loss = {float(loss):.4f}")


def test_gradcam_runs():
    model = build_model("cbam_transformer")
    img = dummy_batch(1)
    last_conv = find_last_conv_layer_name(model)
    heatmap, pred_idx = make_gradcam_heatmap(img, model, last_conv)
    assert heatmap.ndim == 2
    print(f"[OK] Grad-CAM: heatmap shape {heatmap.shape}, predicted class idx {pred_idx}")


if __name__ == "__main__":
    print(f"TensorFlow {tf.__version__}, GPU: {tf.config.list_physical_devices('GPU')}")
    test_all_experiment_architectures()
    test_ssl_forward_pass()
    test_gradcam_runs()
    print("\nAll smoke tests passed — architectures are shape-correct and runnable.")
    print("Real metrics require the actual OASIS images (see data/README_DATA.md) and a GPU.")
