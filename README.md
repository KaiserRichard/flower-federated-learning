# Advanced Federated Learning with Flower & PyTorch

[![Flower](https://img.shields.io/badge/Flower-1.13-orange)](https://flower.ai)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![Weights & Biases](https://img.shields.io/badge/W%26B-Tracking-yellow)](https://wandb.ai/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A production-grade Federated Learning simulation built with **Flower (Flwr)** and **PyTorch**.

This project demonstrates how to implement **custom strategies** to solve complex FL challenges, specifically transmitting **arbitrary metadata** (like training time, loss history, and convergence status) from Client to Server alongside model weights.

## 🚀 Key Features

* **Custom Strategy (`CustomFedAdagrad`)**: Extends the standard `FedAdagrad` strategy to handle custom logic and aggregation.
* **Arbitrary Metadata Transmission**:
    * **Client-Side**: Calculates convergence (loss plateau), packs rich metrics into a dataclass, and serializes it using `pickle`.
    * **Server-Side**: Intercepts the message, deserializes the metadata, and logs it for global analysis.
* **Dynamic Learning Rate**: Implements LR decay (halves every 5 rounds) via the Strategy configuration.
* **Experiment Tracking**: Integrated with **Weights & Biases (W&B)** for real-time visualization of global accuracy and client-specific metrics.
* **Smart Checkpointing**: Automatically saves the global model to disk (`.pth`) whenever a new "best accuracy" is achieved.

## 📂 Project Structure

```text
.
├── my_federated_project
│   ├── client_app.py       # Client logic: Train, Evaluate, Serialize Metadata
│   ├── server_app.py       # Server config, Grid setup, Strategy initialization
│   ├── strategy.py         # Custom Strategy: Metadata deserialization & W&B logging
│   └── task.py             # Shared definitions: Model (Net), Data, Metadata Dataclass
├── outputs/                # Timestamped model checkpoints (Ignored by Git)
├── pyproject.toml          # Simulation configuration
└── README.md

```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/KaiserRichard/flower-federated-learning.git](https://github.com/KaiserRichard/flower-federated-learning.git)
cd flower-federated-learning

```


2. **Install dependencies:**
This project uses `flwr` with simulation support, `torch`, and `wandb`.
```bash
pip install flwr[simulation] torch torchvision flwr-datasets[vision] wandb

```


3. **Log in to Weights & Biases (Optional):**
To enable experiment tracking:
```bash
wandb login

```



## 🏃 Usage

Run the simulation using the Flower CLI. The configuration (rounds, nodes, resources) is managed in `pyproject.toml`.

```bash
flwr run

```

### Configuration

You can adjust parameters in `pyproject.toml`:

```toml
[tool.flwr.app.config]
num-server-rounds = 20
fraction-train = 0.5        # Train on 50% of available nodes per round
local-epochs = 5            # Epochs per client
lr = 0.01                   # Initial Learning Rate

```

## 🧠 Technical Deep Dive: Metadata Passing

One of the biggest challenges in FL is sending data *other* than weights (e.g., "Did my model converge?"). This project solves it using a **Shared Dataclass Contract**:

1. **Define**: A shared class `TrainProcessMetadata` is defined in `task.py`.
2. **Serialize (Client)**: In `client_app.py`, we calculate convergence, create the object, and pickle it:
```python
train_metadata_bytes = pickle.dumps(train_metadata)
config_record = ConfigRecord({"train_metadata": train_metadata_bytes})

```


3. **Deserialize (Strategy)**: In `strategy.py`, we override `aggregate_train` to unpack it:
```python
train_meta = pickle.loads(msg.content["metadata"]["train_metadata"])
print(f"Client Converged: {train_meta.converged}")

```



## 📊 Dashboard & Results

Metrics are tracked automatically. If using W&B, you will see:

* **Global Accuracy & Loss** vs. Rounds
* **Client Training Time** distribution
* **Convergence Events**

## 📜 License

This project is licensed under the Apache 2.0 License.

```

```
