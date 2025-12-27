"""
strategy.py: Custom Strategy inheriting from FedAdagrad.

Features:
- Deserializes custom 'TrainProcessMetadata' from clients.
- Updates Learning Rate dynamically.
- Saves best model checkpoints during training.
- Logs metrics to Weights & Biases (W&B).
"""

import io
import time
from logging import INFO
from pathlib import Path
from typing import Callable, Iterable, Optional
import pickle
from dataclasses import asdict

import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log, logger
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAdagrad, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

PROJECT_NAME = "FLOWER-advanced-pytorch"

class CustomFedAdagrad(FedAdagrad):
    
    def aggregate_train(
            self, 
            server_round: int,
            replies: Iterable[Message],
            failures: Iterable[BaseException]
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """
        Aggregate results and deserialize custom metadata.
        
        Note: We accept 'failures' in the signature because 'start()' sends it,
        but we do NOT pass it to super() because the parent class doesn't support it yet.
        """
        
        # 1. Custom Metadata Logic: Deserialize and Print
        for reply in replies:
            if "metadata" in reply.content and "train_metadata" in reply.content["metadata"]:
                config_record = reply.content["metadata"]
                metadata_bytes = config_record["train_metadata"]
                
                # Deserialize bytes back to DataClass
                train_meta = pickle.loads(metadata_bytes)
                print("TrainProcessMetadata received from client:", asdict(train_meta))

        # 2. Call Parent Aggregation (Standard FedAdagrad logic)
        # Note: We purposely omit 'failures' here to avoid TypeError in current Flwr version
        return super().aggregate_train(server_round, replies)


    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, driver: Grid
    ) -> Iterable[Message]:
        """Configure the next round. Implements Learning Rate Decay."""
        
        # Decrease learning rate by 50% every 5 rounds
        if server_round % 5 == 0 and server_round > 0:
            config["lr"] *= 0.5
            print("LR decreased to:", config["lr"])
            
        return super().configure_train(server_round, arrays, config, driver)

    def set_save_path(self, path: Path):
        """Setter for the output directory path."""
        self.save_path = path

    def _update_best_acc(
        self, current_round: int, accuracy: float, arrays: ArrayRecord
    ) -> None:
        """Save the model to disk if it achieves a new best accuracy."""
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            logger.log(INFO, "💡 New best global model found: %f", accuracy)
            
            file_name = f"model_state_acc_{accuracy}_round_{current_round}.pth"
            torch.save(arrays.to_torch_state_dict(), self.save_path / file_name)
            logger.log(INFO, "💾 New best model saved to disk: %s", file_name)

    def start(
        self,
        driver: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[Callable[[int, ArrayRecord], Optional[MetricRecord]]] = None,
    ) -> Result:
        """
        Main Strategy execution loop.
        Logs to W&B and manages the training/eval cycle.
        """

        if not hasattr(self, 'save_path') or self.save_path is None:
             raise ValueError("You must call set_save_path() before starting the strategy.")

        # Initialize W&B
        name = f"{str(self.save_path.parent.name)}/{str(self.save_path.name)}-ServerApp"
        wandb.init(project=PROJECT_NAME, name=name)

        self.best_acc_so_far = 0.0

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(num_rounds, initial_arrays, train_config, evaluate_config)
        self.summary()
        log(INFO, "")

        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # --- TRAINING ---
            train_replies = driver.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    driver,
                ),
                timeout=timeout,
            )

            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
                failures=[],  # Required by function definition
            )

            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics
                wandb.log(dict(agg_train_metrics), step=current_round)

            # --- EVALUATION ---
            evaluate_replies = driver.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    driver,
                ),
                timeout=timeout,
            )

            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
                # failures=[],  # Not passed to avoid issues with parent
            )

            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics
                wandb.log(dict(agg_evaluate_metrics), step=current_round)

            # --- GLOBAL EVALUATION ---
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                    self._update_best_acc(current_round, res["accuracy"], arrays)
                    wandb.log(dict(res), step=current_round)

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        
        log(INFO, "Final results:")
        log(INFO, "")
        for line in str(result).split("\n"):
            log(INFO, "\t%s", line)
        log(INFO, "")

        return result