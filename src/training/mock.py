from abc import ABC, abstractmethod

class Trainer(ABC):
    @abstractmethod
    def move_batch_to_device(self, batch, device):
        """Move the batch to the specified device."""
        pass

    @abstractmethod
    def compute_loss(self, policy_model, reference_model, batch):
        """Compute the loss for a single batch."""
        pass

    @abstractmethod
    def split_batch(self, batch, start_idx, end_idx):
        """Split a batch into a smaller micro-batch."""
        pass

    @abstractmethod
    def get_batch_size(self, batch):
        """Get the size of the batch."""
        pass