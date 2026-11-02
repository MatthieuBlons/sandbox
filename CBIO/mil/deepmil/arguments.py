from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
import copy
import yaml
from pathlib import Path

def load_config(args):
    """Load YAML config file and update args."""
    if args.config is not None:
        with open(args.config, "r") as f:
            dic = yaml.safe_load(f)
        args.__dict__.update(dic)
    return args


def infer_dataset_info(args):
    """Infer dataset-dependent parameters (e.g., num classes)."""
    target_table = pd.read_csv(args.target_path)
    args.num_class = len(set(target_table[args.target_name]))
    return args


def resolve_training_defaults(args, train):
    """Set training-related defaults."""
    args.train = train
    args.patience = args.epochs if args.patience is None else args.patience
    return args


def configure_lr_scheduler(args):
    """Handle LR scheduler-specific logic."""
    if args.lr_scheduler == "cos":
        args.patience_lr = None

    elif args.lr_scheduler == "linear" and args.patience_lr is None:
        args.patience_lr = args.epochs

    return args


def adjust_features_size(args):
    """Ensure feature dimensions are consistent."""
    assert args.feature_dim <= args.encoder_dim, "feature_dim cannot exceed encoder_dim"

    if args.feature_dim == 0:
        args.feature_dim = args.encoder_dim

    if not args.instance_transf:
        args.feature_depth = args.feature_dim

    if args.feature_depth == 0:
        args.feature_depth = args.feature_dim

    return args


def configure_instance_transform(args):
    """Enable coordinate usage when required by model."""
    if args.instance_transf in ["roformer", "rposbias"]:
        args.use_coords = True
    return args


def configure_sampling_mode(args):
    """Define whether sampling produces constant-size batches."""
    if args.n_tiles == 0 or args.sampler != "random":
        args.constant_size = False
    else:
        args.constant_size = True
    return args


def check_val_metric_sign(args):
    """Define optimization direction for early stopping."""
    if args.ref_metric == "loss":
        args.ref_metric = "mean_val_loss"
        args.sgn_metric = 1
    else:
        args.sgn_metric = -1
    return args


def write_config(args):
    """Write the resolved configuration to the job directory."""
    config_path = Path(args.job_dir) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("w") as file:
        yaml.safe_dump(
            vars(args),
            file,
            sort_keys=False,
            default_flow_style=False,
        )

    return args


# -------------------------------------------------
# Main argument builder
# -------------------------------------------------


def get_arguments(known_args=None, train=True, config=None):
    """
    Build CLI arguments for training / inference pipeline.
    """

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # -------------------------------------------------
    # System / runtime
    # -------------------------------------------------
    parser.add_argument(
        "--device", type=str, default="cpu", help="Compute device (cpu, cuda, mps)"
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Dataloader workers")

    # -------------------------------------------------
    # Config handling
    # -------------------------------------------------
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides CLI args)",
    )
    parser.add_argument(
        "--write_config",
        action="store_true",
        help="Write final config to ./config.yaml",
    )

    # -------------------------------------------------
    # Data paths
    # -------------------------------------------------
    parser.add_argument(
        "--wsi_dir",
        type=str,
        required=True,
        help="Path to WSI feature directory (inputs)",
    )
    parser.add_argument(
        "--target_path", type=str, required=True, help="CSV file containing targets"
    )
    parser.add_argument(
        "--target_name",
        type=str,
        required=True,
        help="Column name of target variable in target CSV",
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        required=True,
        help="Output directory for logs, checkpoints, etc.",
    )

    # -------------------------------------------------
    # WSI encoding
    # -------------------------------------------------
    parser.add_argument(
        "--wsi_format", type=str, default="h5", help="Format of WSI embeddings"
    )
    parser.add_argument(
        "--wsi_enc",
        type=str,
        default="tile",
        help="WSI encoding level (tile, slide, etc.)",
    )
    parser.add_argument(
        "--n_tiles",
        type=int,
        default=0,
        help="Number of tiles per WSI (0 = full slide)",
    )
    parser.add_argument(
        "--use_coords",
        type=bool,
        default=False,
        help="Include tile coordinates in dataloader",
    )
    parser.add_argument(
        "--encoder_dim", type=int, default=1536, help="Embedding dimension from encoder"
    )

    # -------------------------------------------------
    # Sampling
    # -------------------------------------------------
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["all", "random", "random_strict", "niche"],
        default="random",
        help="Tile sampling strategy",
    )
    parser.add_argument(
        "--sample_wr_whole_label",
        type=bool,
        default=True,
        help="Balance sampling across classes",
    )
    parser.add_argument(
        "--no_strat_sampling",
        type=bool,
        default=False,
        help="Disable stratified sampling entirely",
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    parser.add_argument(
        "--model", type=str, default="mhmc", choices=["mhmc"], help="Model architecture"
    )

    parser.add_argument(
        "--feature_dim",
        type=int,
        default=0,
        help="Number of features kept (0 = encoder_dim)",
    )
    parser.add_argument(
        "--feature_depth",
        type=int,
        default=512,
        help="Internal feature dimension after projection",
    )

    parser.add_argument(
        "--instance_transf",
        default=None,
        choices=["linear", "transformer", "roformer", "rposbias"],
        type=str,
        help="Instance-level transformation before pooling",
    )

    parser.add_argument(
        "--pooling_fct",
        type=str,
        default="attn",
        choices=["attn", "attn_max", "attn_gated", "mean", "max"],
        help="Pooling function over tiles",
    )

    parser.add_argument(
        "-k", type=int, default=5, help="Top-k tile selection parameter"
    )

    parser.add_argument(
        "--atn_dim", type=int, default=256, help="Attention hidden dimension"
    )
    parser.add_argument(
        "--num_heads", type=int, default=1, help="Number of attention heads"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.4, help="Dropout probability"
    )

    parser.add_argument(
        "--n_layers_classif", type=int, default=3, help="Number of classifier layers"
    )
    parser.add_argument(
        "--width_fe", type=int, default=512, help="Hidden width of classifier MLP"
    )

    parser.add_argument(
        "--output_layer",
        type=str,
        default="logsoftmax",
        choices=["logsoftmax", "softmax", None],
        help="Output activation function",
    )

    # -------------------------------------------------
    # Validation / training
    # -------------------------------------------------
    parser.add_argument("--use_val", default=1, help="Use validation split")
    parser.add_argument(
        "--val_sampler", default="all", help="Validation sampling strategy"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of WSIs per batch"
    )

    parser.add_argument(
        "--ref_metric",
        type=str,
        default="roc",
        choices=["balanced_acc", "precision", "recall", "f1-score", "roc_auc", "loss"],
        help="Metric used for model selection",
    )

    parser.add_argument("--criterion", type=str, default="nll", help="Loss function")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer type")

    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (default = epochs)",
    )

    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")

    parser.add_argument(
        "--lr_scheduler", type=str, default="cos", help="Learning rate scheduler type"
    )

    parser.add_argument(
        "--patience_lr",
        type=int,
        default=None,
        help="LR scheduler patience (if applicable)",
    )

    # -------------------------------------------------
    # Cross-validation
    # -------------------------------------------------
    parser.add_argument(
        "--k_folds", type=int, default=None, help="Number of cross-validation folds"
    )
    parser.add_argument("--reps", type=int, default=None, help="Repetitions per fold")
    parser.add_argument(
        "--test_fold", type=int, default=0, help="Fold index used as test set"
    )
    parser.add_argument(
        "--repeat", type=int, default=0, help="Repetition index for tracking runs"
    )

    # -------------------------------------------------
    # Inference only
    # -------------------------------------------------
    if not train:
        parser.add_argument(
            "--model_path",
            type=str,
            required=True,
            help="Path to trained model checkpoint",
        )

    # -------------------------------------------------
    # Parse
    # -------------------------------------------------
    args, _ = parser.parse_known_args(known_args)

    # -------------------------------------------------
    # Post-processing pipeline
    # -------------------------------------------------
    args = load_config(args)
    args = infer_dataset_info(args)
    args = resolve_training_defaults(args, train)
    args = configure_lr_scheduler(args)
    args = adjust_features_size(args)
    args = configure_instance_transform(args)
    args = configure_sampling_mode(args)
    args = check_val_metric_sign(args)

    # -------------------------------------------------
    # Save config
    # -------------------------------------------------
    if args.write_config:
        args = write_config(args)

    return args
