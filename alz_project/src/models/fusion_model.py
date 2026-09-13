"""
fusion_model.py  [our contribution — composition layer]

Builds each ablation-study experiment from the shared backbone by toggling components:

  Experiment 1 "baseline"              : Tripathy backbone (I-SAB + multiscale fusion) + MLP
  Experiment 2 "cbam"                  : baseline + CBAM on the final backbone feature map
  Experiment 3 "cbam_transformer"      : + Transformer global-context branch, fused with CNN
  Experiment 4 "cbam_transformer_ssl"  : same architecture as Exp.3, but backbone weights are
                                          initialized from SSL pretraining (see ssl_pretrain.py)
                                          before supervised fine-tuning — the architecture code
                                          is identical, only the initial weights differ, so
                                          build_model(..., experiment="cbam_transformer_ssl")
                                          returns the same graph as "cbam_transformer" and the
                                          SSL step is applied by train.py before compiling.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

from .depthwise_cnn import build_backbone
from .cbam import CBAM
from .transformer_branch import build_transformer_branch
from .. import config as cfg

VALID_EXPERIMENTS = ["baseline", "cbam", "cbam_transformer", "cbam_transformer_ssl"]


def build_model(experiment: str, num_classes: int = None) -> Model:
    if experiment not in VALID_EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{experiment}'. Choose from {VALID_EXPERIMENTS}")
    num_classes = num_classes or cfg.NUM_CLASSES

    inputs = layers.Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.CHANNELS), name="mri_input")
    final_feature_map, multiscale_pooled = build_backbone(inputs, return_multiscale_features=True)

    if experiment == "baseline":
        x = layers.Dense(256, activation="relu", name="mlp_256")(multiscale_pooled)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
        return Model(inputs, outputs, name="experiment1_baseline")

    # Experiments 2-4 all refine the final feature map with CBAM first
    refined = CBAM(name="cbam")(final_feature_map)
    cbam_pooled = layers.GlobalAveragePooling2D(name="cbam_gap")(refined)
    cnn_features = layers.Concatenate(name="cnn_multiscale_plus_cbam")(
        [multiscale_pooled, cbam_pooled]
    )

    if experiment == "cbam":
        x = layers.Dense(256, activation="relu", name="mlp_256")(cnn_features)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
        return Model(inputs, outputs, name="experiment2_cbam")

    # Experiments 3 & 4 (same graph; SSL only changes initial weights, see train.py)
    global_feat = build_transformer_branch(inputs)
    fused = layers.Concatenate(name="cnn_plus_transformer_fusion")([cnn_features, global_feat])
    x = layers.Dense(256, activation="relu", name="mlp_256")(fused)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model_name = "experiment3_cbam_transformer" if experiment == "cbam_transformer" \
        else "experiment4_cbam_transformer_ssl"
    return Model(inputs, outputs, name=model_name)


def get_backbone_submodel(full_model: Model) -> Model:
    """
    Extracts just the CNN-backbone portion of a built model (input -> final conv feature map),
    used by ssl_pretrain.py to attach a projection head for contrastive pretraining, and to
    later transplant pretrained weights back into a fresh fusion model of the same architecture.
    """
    final_conv_layer = None
    for layer in full_model.layers:
        if layer.name.startswith("isab_layer10"):
            final_conv_layer = layer
            break
    if final_conv_layer is None:
        raise RuntimeError("Could not locate final backbone layer 'isab_layer10' in model.")
    return Model(full_model.input, final_conv_layer.output, name="backbone_only")
