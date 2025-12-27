"""
server_app.py: The main entry point for the Server side.

Features:
- Defines the ServerApp.
- Configures the CustomFedAdagrad strategy.
- Manages output directories for checkpoints.
"""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import ServerApp, Grid  
from my_federated_project.task import Net, global_evaluate
from my_federated_project.strategy import CustomFedAdagrad
from datetime import datetime
from pathlib import Path

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Main ServerApp loop.
    
    Args:
        grid: The interface to send messages (replaces 'Driver').
        context: Contains configuration (run_config).
    """

    # Read config from pyproject.toml
    fraction_train: float = context.run_config["fraction-train"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # Initialize Global Model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize Custom Strategy
    strategy = CustomFedAdagrad(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        # Quality Control: Ensure enough nodes participate
        min_train_nodes=5,      
        min_evaluate_nodes=5,   
        min_available_nodes=10, 
    )
    
    # Create Timestamped Output Directory
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Pass the path to the strategy for saving checkpoints
    strategy.set_save_path(save_path)
    
    # Start the Strategy Loop
    result = strategy.start(
        driver=grid, 
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save final model after all rounds
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")