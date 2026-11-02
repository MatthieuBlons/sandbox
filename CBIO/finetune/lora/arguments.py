from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    BooleanOptionalAction,
)
from pathlib import Path

import yaml

# =========================
# Configuration utilities
# =========================


def load_yaml_config(config_path):
    """Load a YAML configuration file."""
    if config_path is None:
        return {}

    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise TypeError(
            "The YAML configuration must contain a mapping of argument names to values."
        )

    return config


def validate_config_keys(parser, config):
    """Reject unknown keys from the YAML configuration."""
    valid_keys = {action.dest for action in parser._actions if action.dest != "help"}

    unknown_keys = set(config) - valid_keys

    if unknown_keys:
        raise ValueError("Unknown configuration arguments: " f"{sorted(unknown_keys)}")

    return config


def write_config(args):
    """Write the resolved configuration to the job directory."""
    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    config_path = job_dir / "config.yaml"

    with config_path.open("w") as file:
        yaml.safe_dump(
            vars(args),
            file,
            sort_keys=False,
            default_flow_style=False,
        )

    return args


# =========================
# Argument configuration
# =========================


def resolve_training_defaults(args, train):
    """Set training-related defaults."""
    args.train = train

    if args.patience is None:
        args.patience = args.epochs

    return args


def configure_lr_scheduler(args):
    """Resolve learning-rate scheduler-dependent parameters."""
    if args.lr_scheduler in {"none", "cos"}:
        args.patience_lr = None

    elif args.lr_scheduler == "linear" and args.patience_lr is None:
        args.patience_lr = args.epochs

    elif args.lr_scheduler == "plateau" and args.patience_lr is None:
        args.patience_lr = max(1, args.patience // 2)

    return args


def configure_encoder(args):
    """Resolve encoder-dependent parameters."""
    if not args.lora:
        args.lora_r = None
        args.lora_a = None
        args.lora_target = []
    return args


def configure_feature_dimensions(args):
    """Resolve feature dimensions before classification."""
    if args.feature_dim <= 0:
        raise ValueError("feature_dim must be greater than zero.")

    if args.feature_depth < 0:
        raise ValueError("feature_depth cannot be negative.")

    if not args.instance_transf:
        args.feature_depth = args.feature_dim

    elif args.feature_depth == 0:
        args.feature_depth = args.feature_dim

    return args


def configure_classifier_layers(args):
    """
    Normalize old and current classifier configurations.

    Old configuration:
        width_fe: int
        n_layers_classif: int

    Current configuration:
        width_fe: list[int]

    Infer classifier depth from the hidden-width specification.
    """

    # Legacy checkpoints stored a single width and a separate layer count.
    if isinstance(args.width_fe, int):
        n_layers = getattr(args, "n_layers_classif")
        args.width_fe = [args.width_fe] * n_layers

    # Current configuration already stores each hidden-layer width in a list.
    elif isinstance(args.width_fe, (list, tuple)):
        args.n_layers_classif = len(args.width_fe)

    return args


def configure_validation_metric(args):
    """Define the optimization direction for model selection."""
    metric_mapping = {
        "loss": ("mean_val_loss", -1),
        "pearsonr": ("mean_pearsonr", 1),
        #"cossim": ("mean_spearmanr", 1),
        #"euclidiansim": ("mean_euclidsim", 1),
        #"euclidiandist": ("mean_eucliddist", -1),
        #"spearmanr": ("mean_spearmanr", 1),
        #"r2": ("r2", 1),
    }

    args.ref_metric, args.sgn_metric = metric_mapping[args.ref_metric]

    return args


# =========================
# Argument validation
# =========================


def validate_cross_validation(args):
    """Validate fold and repetition indices."""
    if args.k_folds is not None:
        if args.k_folds < 2:
            raise ValueError("k_folds must be at least 2.")

        if not 0 <= args.test_fold < args.k_folds:
            raise ValueError(
                f"test_fold must be between 0 and {args.k_folds - 1}, "
                f"received {args.test_fold}."
            )

    elif args.test_fold != 0:
        raise ValueError("test_fold was provided without setting k_folds.")

    if args.reps is not None:
        if args.reps < 1:
            raise ValueError("reps must be at least 1.")

        if not 0 <= args.repeat < args.reps:
            raise ValueError(
                f"repeat must be between 0 and {args.reps - 1}, "
                f"received {args.repeat}."
            )

    elif args.repeat != 0:
        raise ValueError("repeat was provided without setting reps.")

    return args


# =========================
# Parser construction
# =========================


def build_parser(train=True):
    """Build the complete command-line parser."""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # -------------------------------------------------
    # Configuration
    # -------------------------------------------------
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file",
    )
    parser.add_argument(
        "--write_config",
        action="store_true",
        help="Write the resolved configuration to job_dir/config.yaml",
    )

    # -------------------------------------------------
    # Data paths
    # -------------------------------------------------
    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        help="Directory containing the H&E tiles used as inputs",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Path to the table containing the data split",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default=None,
        help="Path to the HDF5 file containing proteomic signatures",
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        default=None,
        help="Directory used to store outputs",
    )

    # -------------------------------------------------
    # Runtime
    # -------------------------------------------------
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers",
    )

    # -------------------------------------------------
    # Cross-validation
    # -------------------------------------------------
    parser.add_argument(
        "--k_folds",
        type=int,
        default=None,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--test_fold",
        type=int,
        default=0,
        help="Index of the fold used as the test set",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=None,
        help="Number of repetitions per fold",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="Repetition index used to identify the run",
    )

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------
    parser.add_argument(
        "--use_val",
        action=BooleanOptionalAction,
        default=True,
        help="Use a validation set during training",
    )
    parser.add_argument(
        "--ref_metric",
        type=str,
        choices=["loss", "pearsonr", "spearmanr", "r2"],
        default="loss",
        help="Validation metric used for model selection",
    )

    # -------------------------------------------------
    # Optimization
    # -------------------------------------------------
    parser.add_argument(
        "--criterion",
        type=str,
        choices=["hierachy", "mse", "rmse", "mae", "l1", "huber"],
        default="mse",
        help="Training loss",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw", "sgd"],
        default="adamw",
        help="Optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Optimizer weight decay",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-3,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early-stopping patience; defaults to epochs",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["none", "cos", "linear", "plateau"],
        default="cos",
        help="Learning-rate scheduler",
    )
    parser.add_argument(
        "--patience_lr",
        type=int,
        default=None,
        help="Scheduler patience or linear-decay duration",
    )

    # -------------------------------------------------
    # Encoder
    # -------------------------------------------------
    parser.add_argument(
        "--encoder",
        type=str,
        default="hoptimus1",
        help="Image encoder",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Input image size",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=1536,
        help="Encoder output dimension",
    )

    # -------------------------------------------------
    # LoRA
    # -------------------------------------------------
    parser.add_argument(
        "--lora",
        action=BooleanOptionalAction,
        default=True,
        help="Fine-tune the encoder using LoRA",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_a",
        type=float,
        default=1.0,
        help="LoRA scaling factor alpha",
    )
    parser.add_argument(
        "--lora_target",
        type=str,
        nargs="+",
        default=["qkv"],
        help="Encoder modules targeted by LoRA",
    )

    # -------------------------------------------------
    # Regression head
    # -------------------------------------------------
    parser.add_argument(
        "--instance_transf",
        action=BooleanOptionalAction,
        default=True,
        help="Transform encoder features before classification",
    )
    parser.add_argument(
        "--feature_depth",
        type=int,
        default=512,
        help="Feature dimension after the optional projection",
    )
    parser.add_argument(
        "--width_fe",
        type=int,
        nargs="*",
        default=[],
        help="Hidden dimensions of the prediction MLP",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout probability",
    )
    parser.add_argument(
        "--output_lenght",
        type=int,
        default=1,
        help="target lenght",
    )
    # -------------------------------------------------
    # Dataloader
    # -------------------------------------------------
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of image tiles per batch",
    )

    # -------------------------------------------------
    # Inference
    # -------------------------------------------------
    if not train:
        parser.add_argument(
            "--model_path",
            type=str,
            default=None,
            help="Path to the trained model checkpoint",
        )

    return parser


# =========================
# Parsing pipeline
# =========================


def get_arguments(known_args=None, train=True, config=None):
    """
    Parse command-line and YAML arguments.

    Precedence:
        parser defaults < YAML configuration < explicit CLI arguments
    """
    # Parse only the configuration path first.
    config_parser = ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=str,
        default=config,
    )

    config_args, _ = config_parser.parse_known_args(known_args)
    yaml_config = load_yaml_config(config_args.config)

    # Build the complete parser.
    parser = build_parser(train=train)
    validate_config_keys(parser, yaml_config)

    # YAML values replace parser defaults.
    parser.set_defaults(**yaml_config)

    # Explicit CLI values replace YAML values.
    args = parser.parse_args(known_args)

    # Required arguments are validated after merging YAML and CLI values.
    required_arguments = [
        "img_dir",
        "input_path",
        "target_path",
        "job_dir",
    ]

    if not train:
        required_arguments.append("model_path")

    missing_arguments = [
        argument
        for argument in required_arguments
        if getattr(args, argument, None) is None
    ]

    if missing_arguments:
        parser.error(
            "Missing required arguments after resolving YAML and CLI values: "
            + ", ".join(f"--{argument}" for argument in missing_arguments)
        )

    # Post-processing.
    args = resolve_training_defaults(args, train=train)
    args = configure_lr_scheduler(args)
    args = configure_encoder(args)
    args = configure_feature_dimensions(args)
    args = configure_classifier_layers(args)
    args = configure_validation_metric(args)
    args = validate_cross_validation(args)

    if args.write_config:
        args = write_config(args)

    return args
