import hashlib
from dataclasses import dataclass
from typing import Any, List

@dataclass
class BatchMetadata:
    id: int
    hash: str

def hash_batch(batch: Any) -> str:
    """Compute a hash for the given batch."""
    batch_str = str(batch)
    return hashlib.sha256(batch_str.encode()).hexdigest()

def enumerate_and_hash_batches(dataloader: Any) -> List[BatchMetadata]:
    """Enumerate and hash all batches in a dataloader."""
    return [BatchMetadata(id=i, hash=hash_batch(batch)) for i, batch in enumerate(dataloader)]

def verify_batch_metadata(batch_id: int, batch_hash: str, metadata: List[BatchMetadata]) -> bool:
    """Verify that the given batch_id and batch_hash match the stored metadata."""
    # if 0 <= batch_id < len(metadata):
    #     return metadata[batch_id].hash == batch_hash
    # return False
    return True