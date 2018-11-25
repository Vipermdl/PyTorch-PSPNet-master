from argparse import ArgumentParser


def get_arguments():
    """Defines command-line arguments, and parses them.

    """
    parser = ArgumentParser()

    # Execution mode
    parser.add_argument(
        "--mode",
        "-m",
        choices=['train', 'test', 'full'],
        default='train',
        help=("train: performs training and validation; test: tests the model "
              "found in \"--save_dir\" with name \"--name\" on \"--dataset\"; "
              "full: combines train and test modes. Default: train"))
    parser.add_argument(
        "--resume",
        dest='resume',
        default=0, type=int,
        help="The model found in \"--checkpoint_dir/--name/\" and filename "
              "\"--name.h5\" is loaded.")
    
    # Hyperparameters
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="The batch size. Default: 10")
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs. Default: 300")
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=5e-4,
        help="The learning rate. Default: 5e-4")
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.1,
        help="The learning rate decay factor. Default: 0.5")
    parser.add_argument(
        "--lr-decay-epochs",
        type=int,
        default=100,
        help="The number of epochs before adjusting the learning rate. "
        "Default: 100")
    parser.add_argument(
        "--weight-decay",
        "-wd",
        type=float,
        default=2e-4,
        help="L2 regularization factor. Default: 2e-4")

    # Dataset
    parser.add_argument(
        "--dataset",
        choices=['camvid', 'cityscapes'],
        default='cityscapes',
        help="Dataset to use. Default: camvid")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="../dataset/cityscape",
        help="Path to the root directory of the selected dataset. "
        "Default: ../dataset/cityscapes")
    parser.add_argument(
        "--height",
        type=int,
        default=360,
        help="The image height. Default: 360")
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="The image height. Default: 480")
    parser.add_argument(
        "--weighing",
        choices=['net', 'mfb', 'none'],
        default='net',
        help="The class weighing technique to apply to the dataset. "
        "Default: PSPNet")
    parser.add_argument(
        "--with-unlabeled",
        dest='ignore_unlabeled',
        action='store_false',
        help="The unlabeled class is not ignored.")

    # Settings
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. Default: 0")
    parser.add_argument(
        "--print-step",
        type=int,
        default=1,
        help="Print loss every step")
    parser.add_argument(
        "--imshow-batch",
        type=int,
        default=0,
        help=("Displays batch images when loading the dataset and making "
              "predictions."))
    parser.add_argument(
        "--cuda",
        dest='cuda',
        type=int,
        default=1,
        help="whether use CUDA.")
    parser.add_argument(
        "--visdom",
        dest='visdom',
        default=1,type=int,
        help="whether use visdom.")
    

    # Storage settings
    parser.add_argument(
        "--name",
        type=str,
        default='PSPNet',
        help="Name given to the model when saving. Default: PSPNet")
    parser.add_argument(
        "--save-dir",
        type=str,
        default='save',
        help="The directory where models are saved. Default: save")

    return parser.parse_args()
