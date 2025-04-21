import argparse
import logging
import os
import sys
import uuid
import time
from pathlib import Path
import train

logger = logging.getLogger(__name__)

def parse_args():
    trainer_parser = train.get_args_parser()
    parser = argparse.ArgumentParser(
        "Submitit for flow_matching training", parents=[trainer_parser]
    )
    parser.add_argument(
        "--ngpus", default=8, type=int, help="Number of gpus to request on each node"
    )
    parser.add_argument(
        "--nodes", default=8, type=int, help="Number of nodes to request"
    )
    parser.add_argument("--timeout", default=4320, type=int, help="Duration of the job")
    parser.add_argument(
        "--job_dir", default="", type=str, help="Job dir. Leave empty for automatic."
    )
    parser.add_argument(
        "--shared_dir",
        default="/content/shared_dir",  # A directory within Colab's file system
        type=str,
        help="Directory shared among the nodes. A directory named USER/experiments is created under shared_dir that is used to coordinate in distributed mode.",
    )
    parser.add_argument(
        "--partition", default="learnlab", type=str, help="Partition where to submit"
    )
    parser.add_argument(
        "--constraint",
        default="",
        type=str,
        help="Slurm constraint eg.: ampere80gb For using A100s or volta32gb for using V100s.",
    )
    parser.add_argument(
        "--comment", default="", type=str, help="Comment to pass to scheduler"
    )
    parser.add_argument("--qos", default="", type=str, help="Slurm QOS")
    parser.add_argument("--account", default="", type=str, help="Slurm account")
    parser.add_argument(
        "--exclude",
        default="",
        type=str,
        help="Exclude certain nodes from the slurm job.",
    )
    return parser.parse_args()


def get_shared_folder(shared_dir: str) -> Path:
    base_path = Path(shared_dir)
    if not base_path.is_dir():
        base_path.mkdir(parents=True, exist_ok=True)

    user = os.getenv("USER")

    if user is None:
        user = "colab_user"  # A default user for Colab

    p = base_path / user / "experiments"
    p.mkdir(exist_ok=True)
    return p


def get_init_file(shared_dir: str):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder(shared_dir)), exist_ok=True)
    init_file = get_shared_folder(shared_dir) / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import train

        self._setup_gpu_args()

        # Convert job_dir to string before passing to train.main
        self.args.job_dir = str(self.args.job_dir)
        train.main(self.args)

    def checkpoint(self):
        self.args.dist_url = get_init_file(self.args.shared_dir).as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")

        if os.path.exists(checkpoint_file) and not self.args.eval_only:
            self.args.resume = checkpoint_file

        logger.info("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)

        return empty_trainer # Modify to return the trainer instance directly

    def _setup_gpu_args(self):
        self.args.log_dir = self.args.output_dir
        self.args.gpu = 0 # Set to 0 for single GPU in Colab
        self.args.rank = 0 # Set to 0 for single process in Colab
        self.args.world_size = 1 # Set to 1 for single process in Colab

        logger.info(
            f"Process group: {self.args.world_size} tasks, rank: {self.args.rank}"
        )


def main():
    args = parse_args()
    if args.job_dir == "":
        shared_folder = get_shared_folder(args.shared_dir)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.job_dir = shared_folder / f"colab_run_{timestamp}"
    else:
        args.job_dir = Path(args.job_dir)

    # Ensure the job directory (output directory) exists
    os.makedirs(args.job_dir, exist_ok=True)
    args.output_dir = str(args.job_dir)  # Convert PosixPath to string
    args.log_dir = str(args.job_dir)     # Convert PosixPath to string

    trainer = Trainer(args)
    trainer() # Directly call the trainer


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
