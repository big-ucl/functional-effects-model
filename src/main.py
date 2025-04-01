from parser import parse_cmdline_args
import torch
from train import train


def main(raw_args = None):
    """
    Main function to run the depression scale model.

    Parameters
    ----------
    raw_args : list, optional
        List of command line arguments. If None, the function will use the
        default arguments defined in the parser.
    """
    #parse command line arguments
    args = parse_cmdline_args(raw_args=raw_args)

    # Set the seed for reproducibility
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Train the model
    train(args=args)

