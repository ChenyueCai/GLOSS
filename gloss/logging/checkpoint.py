# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging and checkpoint management.
"""

from typing import Any, Dict, Mapping, Optional
import os
import pathlib
import logging
import sys

import datetime
from multiprocessing import Process, Queue

from omegaconf import OmegaConf
import torch
import imageio

from gloss.config import ExperimentConfig
from gloss.logging import get_class_logger

class Checkpoint:
    """Checkpointing class.
    
    Handles object logging and model checkpointing.
    """
    # Hard-coding where the checkpoints are saved on PBSS.
    S3_CHECKPOINT_CONTAINER = "checkpoints"

    def __init__(
        self,
        cfg: ExperimentConfig,
        ckpt_path: Optional[os.PathLike] = None,
        global_rank: Optional[int] = None,
        use_s3: bool = False
    ):
        """Initialize the checkpoint object.

        Args:
            cfg (ExperimentConfig): The experiment config.
            ckpt_path (Optional[os.PathLike], optional): Optional path to a checkpoint directory.
                Defaults to None, and a new directory is created.
            global_rank (Optional[int], optional): The global rank of the process owning this
                checkpoint. Defaults to None.
            use_s3 (bool, optional): If true, enabled capabilities to upload/download checkpoints
                from PBSS/PDX. Assumes that s5cmd is initialized for the shell running the
                process (usually this is handled by setting up a .env file). Defaults to False.
        """
        self.cfg = cfg
        self.use_s3 = use_s3

        self._ss_container = Checkpoint.S3_CHECKPOINT_CONTAINER

        ## Initialize the checkpoint directory
        if ckpt_path is None:
            # Create a new checkpoint path with current timestamp
            time_stamp = datetime.datetime.now().strftime("_%b%d%y_%H%M%S")
            outdir = cfg.exp_root_dir
            self.save_dir = os.path.join(
                outdir,
                cfg.project_subgroup,
                f"{cfg.project_name}{time_stamp}"
            )
        else:
            self.save_dir = ckpt_path
            if os.path.exists(self.save_dir):
                print(f"Continuing from checkpoint {pathlib.Path(self.save_dir).stem}...")
        os.makedirs(self.save_dir, exist_ok=True)

        ## Set up logging
        log_path = self.get_new_log_path(global_rank)
        handlers = [logging.FileHandler(log_path)]

        if global_rank is None or global_rank == 0:
            # The rank 0 object should also print to console
            print("Printing to stdout")
            handlers.append(logging.StreamHandler(sys.stdout))
            # Rank 0 also saves the config
            with open(self.get_path('config.yaml'), "w", encoding="utf-8") as f:
                OmegaConf.save(config=cfg, f=f)
        # Set the logging config for adoption elsewhere in code
        logging.basicConfig(
            format="[%(levelname)s:%(asctime)s:%(name)s]:%(message)s",
            handlers=handlers,
            level=logging.INFO
        )
        # Instantiate the logger for this class
        self.logger = get_class_logger(self)

        # Multi-threaded object saving state
        self.n_processes = 4
        self._saving_active = False
        self._save_queue = None
        self._processes = None


    def get_path(self, *args: str) -> pathlib.Path:
        """Get path within the save directory

        # Example
        ```python
        save_path = self.get_path('my', 'folder', 'file.txt')
        print(save_path) # prints: {self._save_dir}/my/folder/file.txt
        ```

        Returns:
            pathlib.Path: Constructed path within the save directory
        """
        return pathlib.Path(self.save_dir, *args)


    def _get_log_path(self, idx=0, rank=None) -> pathlib.Path:
        """Get path to the logging file, with added suffices for current index and rank.


        Args:
            idx (int, optional): Incremental index for resuming runs. Defaults to 0.
            rank (_type_, optional): Rank of this process. Defaults to None, removing
                the rank suffix from the log path.

        Returns:
            pathlib.Path: Path to the log file.
        """
        if rank is None:
            log_fname = f"log_part{idx}.txt"
        else:
            log_fname = f"log_rank{rank}_part{idx}.txt"
        return self.get_path(log_fname)


    def get_new_log_path(self, rank: Optional[int] = None) -> pathlib.Path:
        """Gets an unused log file path.

        This function increments a suffix on the log file until
        an unused file path is found. This path is then returned.

        Args:
            rank (int, optional): The rank of this process. Defaults to None.

        Raises:
            RuntimeError: No unused file path found (up to 1000 tried).

        Returns:
            pathlib.Path: Path to the log file.
        """
        i = 0
        while i < 1000:
            log_path = self._get_log_path(i, rank)
            # We might have a log file here already
            # e.g. if we're restarting training
            # We'll start a new log file to separate these
            if os.path.exists(log_path):
                i += 1
                exists = True
            else:
                exists = False
                break
        if exists:
            raise RuntimeError(
                "Unable to find a new log file.\
                This suggests the max of 1000 was reached. An error?"
            )
        return log_path

    def save_trainer_state(
            self,
            state: Mapping[str, Any],
            fname: str = "latest",
            upload_s3: bool = False
    ) -> str:
        """Save the trainer state with the given file name.

        Args:
            state (Mapping[str, Any]): State dictionary to save.
            fname (str, optional): File name to save the state under.
                Defaults to "latest".
            upload_s3 (bool): If true, upload the checkpoint to PBSS too. Defaults to False.
        Returns:
            path (str): Constructed path string within the save directory
        """
        # Ensure the checkpoints directory exists
        os.makedirs(self.get_path("checkpoints"), exist_ok=True)
        # Construct the path
        path = self.get_path("checkpoints", f"{fname}.pth")
        # Save the state
        torch.save(state, path)
        # If we're using S3, upload the checkpoint as well using s5cmd
        # TODO: ERROR HANDLING
        if self.use_s3 and upload_s3:
            os.system(f"s5cmd cp {str(path)} s3://{self._ss_container}/{path}")
        return str(path)


    def load_trainer_state(
        self,
        path: Optional[pathlib.Path] = pathlib.Path("latest"),
        download_s3: bool = False,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Load the trainer state at the given path.

        Args:
            path (pathlib.Path, optional): Path to the saved state (relative to the save directory).
                Defaults to None.
            download_s3 (bool, optional): If true, try to download the state from PBSS first.
                Defaults to False.
            device (Optional[torch.device], optional): CUDA device to load the state onto.
                Defaults to None.

        Raises:
            ValueError: No checkpoint found at the given path.

        Returns:
            Dict[str, Any]: The state dictionary.
        """
        if path is None:
            path = self.get_path("checkpoints", "latest.pth")
        # TODO: ERROR HANDLING
        if self.use_s3 and download_s3:
            os.system(f"s5cmd cp s3://{self._ss_container}/{path} {path}")
        if os.path.exists(path):
            return torch.load(path, map_location=device)
        raise ValueError(f"No checkpoint found at {path}")


    def start_save_processes(self):
        """Start the worker threads used for object saving.

        Raises:
            RuntimeError: If processes are still running but saving inactive.
        """
        # If we're already saving, then skip
        if self._saving_active:
            return
        # Check that we don't already have processes running. If we do, something went wrong
        # Raise an error
        if self._processes is not None:
            raise RuntimeError("Processes still running. Cannot start new save process")
        # Flag that saving is active
        self._saving_active = True
        # Initialize the save queue
        self._save_queue = Queue()
        # Spin up threads to watch the queue and grab items + save them to disk
        def watch_and_save(queue):
            while True:
                if not queue.empty():
                    filename, data = queue.get()
                    if filename is None:
                        break
                    fpath = pathlib.Path(filename)
                    ext = fpath.suffix

                    # Check if extension is an image
                    if ext.lower() in ['.png', '.jpg', '.jpeg']:
                        imageio.imwrite(fpath, data.cpu().squeeze().numpy())
                    else:
                        torch.save(data, fpath)
        self._processes = [
            Process(target=watch_and_save, args=(self._save_queue,))
            for _ in range(self.n_processes)
        ]
        # Start the processes
        for p in self._processes:
            p.start()


    def save_tensor(self, fname: str, data: torch.Tensor):
        """Save tensor with the given file name.

        Args:
            fname (str): Filename of this tensor.
            data (torch.Tensor): Tensor to save.

        Raises:
            ValueError: Save processes have not been started.
        """
        if self._saving_active and self._save_queue is not None:
            # We make the directory here for over-cautious thread safety
            fpath = self.get_path(os.path.join("outputs", fname))
            os.makedirs(pathlib.Path(fpath).parent, exist_ok=True)
            self._save_queue.put((fpath, data))
        else:
            e = ValueError("Save processes must be started before tensor can be saved.")
            self.logger.error('Error saving tensor', exc_info=e)
            raise e


    def save_image(self, fname: str, image: torch.Tensor):
        """Save image with the given file name.

        Args:
            fname (str): Filename of this image.
            image (torch.Tensor): Image tensor to save.

        Raises:
            ValueError: Filename extension must be '.png', '.jpg', or '.jpeg'.
            ValueError: Save processes have not been started.
        """
        fpath = pathlib.Path(fname)
        ext = fpath.suffix

        # Check if extension is an image
        if ext.lower() not in ['.png', '.jpg', '.jpeg']:
            e = ValueError("Image file extension must be .png, .jpg, or .jpeg.")
            self.logger.error('Error saving image', exc_info=e)
            raise e
        if self._saving_active and self._save_queue is not None:
            # We make the directory here for over-cautious thread safety
            fpath = self.get_path(os.path.join("outputs", fname))
            os.makedirs(pathlib.Path(fpath).parent, exist_ok=True)
            self._save_queue.put((fpath, image))
        else:
            e = ValueError("Save processes must be started before image can be saved.")
            self.logger.error('Error saving image', exc_info=e)
            raise e


    def stop_save_processes(self):
        """Deactivate the save processes.
        """
        self._saving_active = False
        # Check until the queue is empty and then kill all processes
        if self._save_queue is not None and self._processes is not None:
            # Inject None entries to kill the processes
            for _i in range(len(self._processes)):
                self._save_queue.put((None, None))

            # Join processes to terminate when complete
            for p in self._processes:
                p.join()
            self._processes = None
