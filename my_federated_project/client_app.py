"""
client_app.py: Defines the ClientApp logic.

Responsibilities:
1. Receive global model weights.
2. Train on local data partition.
3. Calculate custom metrics (Convergence).
4. Serialize metadata and return updates to the Server.
"""

import pickle
import time
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, ConfigRecord
from flwr.clientapp import ClientApp

from my_federated_project.task import Net, load_data
from my_federated_project.task import test as test_fn
from my_federated_project.task import train as train_fn
from my_federated_project.task import get_device, TrainProcessMetadata

# Initialize ClientApp
app = ClientApp()
device = get_device()

@app.train()
def train(msg: Message, context: Context):
    """
    Train the model on local data.
    """
    # 1. Initialize Model & Load Global Weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(device)

    # 2. Load Local Data Partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)

    # 3. Start Training
    start_time = time.time()

    # Note: We capture both the final loss and the history of losses
    train_loss, epoch_losses_dict = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    end_time = time.time()
    training_time = end_time - start_time

    # 4. Custom Logic: Determine Convergence
    # If the loss difference between the last two epochs is smaller than 0.001,
    # we consider the model 'converged' locally.
    losses_list = list(epoch_losses_dict.values())
    if len(losses_list) > 1:
        last_loss = losses_list[-1]
        second_last_loss = losses_list[-2]
        loss_diff = abs(last_loss - second_last_loss)
        is_converged = loss_diff < 0.001
    else: 
        is_converged = False 

    # 5. Create Metadata Object
    train_metadata = TrainProcessMetadata(
        training_time=training_time,
        converged=is_converged,
        training_losses=epoch_losses_dict,
    )

    # 6. Serialization
    # We use pickle to convert the custom dataclass into bytes so it can be 
    # transported inside the standard ConfigRecord.
    train_metadata_bytes = pickle.dumps(train_metadata)
    config_record = ConfigRecord({"train_metadata": train_metadata_bytes})

    # 7. Construct Reply Message
    model_record = ArrayRecord(model.state_dict())
    
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)

    # Return everything in RecordDict
    content = RecordDict({
        "arrays": model_record,
        "metrics": metric_record,
        "metadata": config_record, # <--- Sending the custom metadata here
    })
    
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions)

    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)