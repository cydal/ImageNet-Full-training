"""
Distributed training utilities.
Helpers for multi-node training setup and environment variable reading.
"""
import os
import socket


def setup_distributed_env():
    """
    Setup distributed training environment variables.
    Reads from common environment variables set by torchrun or SLURM.
    """
    # torchrun sets these automatically, but we can override if needed
    
    # RANK: Global rank of the process
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    
    # LOCAL_RANK: Rank of the process on the current node
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    
    # WORLD_SIZE: Total number of processes
    if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    
    # MASTER_ADDR: Address of the master node
    if "MASTER_ADDR" not in os.environ:
        if "SLURM_NODELIST" in os.environ:
            # Parse first node from SLURM nodelist
            nodelist = os.environ["SLURM_NODELIST"]
            # Simple parsing (works for single node or node[001-004] format)
            master_node = nodelist.split(",")[0].split("[")[0]
            os.environ["MASTER_ADDR"] = master_node
        else:
            os.environ["MASTER_ADDR"] = "localhost"
    
    # MASTER_PORT: Port for communication
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    
    # NODE_RANK: Rank of the current node
    if "NODE_RANK" not in os.environ and "SLURM_NODEID" in os.environ:
        os.environ["NODE_RANK"] = os.environ["SLURM_NODEID"]


def get_rank() -> int:
    """Get global rank of current process."""
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    """Get local rank of current process (rank within node)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """Get total number of processes."""
    return int(os.environ.get("WORLD_SIZE", 1))


def get_node_rank() -> int:
    """Get rank of current node."""
    return int(os.environ.get("NODE_RANK", 0))


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    return get_rank() == 0


def get_hostname() -> str:
    """Get hostname of current node."""
    return socket.gethostname()


def print_distributed_info():
    """Print distributed training information."""
    if is_main_process():
        print("=" * 60)
        print("Distributed Training Info:")
        print(f"  Hostname: {get_hostname()}")
        print(f"  World Size: {get_world_size()}")
        print(f"  Node Rank: {get_node_rank()}")
        print(f"  Master Addr: {os.environ.get('MASTER_ADDR', 'N/A')}")
        print(f"  Master Port: {os.environ.get('MASTER_PORT', 'N/A')}")
        print("=" * 60)
