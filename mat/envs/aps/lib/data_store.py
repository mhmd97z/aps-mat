import torch
from collections import deque


class DataStore:
    def __init__(self, T, keys):
        # Initialize deques with a max length of T for each key
        self.T = T
        self.data = {key: deque(maxlen=T) for key in keys}
    
    def add(self, **kwargs):
        # Add data to the respective deques based on provided key-value pairs
        for key, value in kwargs.items():
            if key in self.data:
                self.data[key].append(value)
            else:
                raise KeyError(f"Key '{key}' is not in the data structure.")

    def get_last_k_elements(self, k=None):
        if k is None:
            k = self.T
        result = {}
        for key in self.data:
            current_data = list(self.data[key])
            if current_data[0] is None: # for example, where there is no gnn embedding
                continue
            if len(current_data) < k:
                # Pad with zeros if fewer than k elements are available
                zeros = torch.zeros_like(current_data[0])
                # Repeat the zero tensor to fill the missing entries
                padded_data = torch.cat([
                    zeros.unsqueeze(0).repeat(k - len(current_data), *([1] * zeros.dim())),  # Repeat zeros in the batch dimension
                    torch.stack(current_data)
                ])
            else:
                if isinstance(current_data[-1], torch.Tensor):
                    padded_data = torch.stack(current_data[-k:])
                else:
                    padded_data = current_data[-k:]
            result[key] = padded_data
        return result

    def __str__(self):
        # Return a string representation of the underlying data
        result = "\n"
        for key, deque_data in self.data.items():
            result += f"{key}: {list(deque_data)}\n"
        return result

    def __len__(self):
        # Return the length of any deque (they all should have the same length)
        if self.data:
            # Assuming all deques have the same length
            return len(next(iter(self.data.values())))
        return 0  # If there's no data
