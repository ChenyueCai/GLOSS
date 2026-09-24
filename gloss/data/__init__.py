# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data loading and handling.
"""
from abc import ABC, abstractmethod
from typing import List

from torch.utils.data import dataloader

from gloss.config import ExperimentConfig
from gloss.logging import get_class_logger


class DataCollection(ABC):
    """An abstract dataset collection.
    
    Provides a standard interface for building a training data loader, and test data loaders.
    """

    def __init__(self, cfg: ExperimentConfig):
        """Initialize the collection using the experiment config.

        Args:
            cfg (ExperimentConfig): The experiment config.
            rank (Optional[int]): The rank of the running process.
        """
        self._logger = get_class_logger(self)
        self._cfg = cfg
        self._train_loader = self.initialize_train_loader()
        self._test_loaders = self.initialize_test_loaders()

    @property
    def train_loader(self) -> dataloader.DataLoader:
        """The train loader.

        Returns:
            dataloader.DataLoader: The training dataset loader.
        """
        return self._train_loader

    @property
    def test_loaders(self) -> List[dataloader.DataLoader]:
        """The test loaders.

        Returns:
            dataloader.DataLoader: The list of test dataset loaders.
        """
        return self._test_loaders

    def _validate_tags(self, datatags: List[str]):
        # Check that we don't mix WDS tags with others
        use_wds = False
        for d in datatags:
            if d.startswith("WDS"):
                use_wds = True
        for d in datatags:
            if use_wds and not d.startswith("WDS"):
                e = ValueError("Cannot mix and match WDS datasets with other types")
                self._logger.error('Error when initializing DataCollection', exc_info=e)
                raise e

    def initialize_train_loader(self) -> dataloader.DataLoader:
        """Initialize the training data loader.

        The base implementation of this method does some simple validation on the data tags
        and then calls `process_train_datatags`.

        Returns:
            dataloader.DataLoader: The training data loader, returned by `process_train_datatags`.
        """
        train_datatags = self._cfg.dataset.train_sets
        self._validate_tags(train_datatags)
        self._logger.info("Using training dataset(s):\n%s", ',\n'.join(train_datatags))
        # Can support other dataset loaders here
        return self.process_train_datatags(train_datatags)

    @abstractmethod
    def process_train_datatags(self, train_datatags: List[str]) -> dataloader.DataLoader:
        """Process the train datatags to create the data loader.

        All concrete instances of this class should provide an implementation of this method.
        The implementation is responsible for parsing the data tags and producing the final
        data loader from them.

        Args:
            train_datatags (List[str]): A list of data tags that will concatenated into a
                training data loader. For example:
                    [
                        'WDS:s3://shard_set_0/',
                        'WDS:s3://shard_set_1/',
                        'WDS:/path/to/local_shard_set/'
                    ]

        Returns:
            dataloader.DataLoader: _description_
        """


    def initialize_test_loaders(self) -> List[dataloader.DataLoader]:
        """Initialize the test data loaders.

        Returns:
            List[dataloader.DataLoader]: An optional list of data loaders
                for model evaluation. By default, returns an empty list.
        """
        return []
