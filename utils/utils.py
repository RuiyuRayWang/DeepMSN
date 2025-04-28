import torch

def one_hot_encode(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1],
               'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1]}
    one_hot_encoded = [mapping[base] for seq in sequence for base in seq]
    return torch.tensor(one_hot_encoded, dtype=torch.float)

def decode_one_hot(one_hot_sequence):
    mapping = ['A', 'C', 'G', 'T']
    if one_hot_sequence.dim() == 2:  # Single sequence
        decoded_sequence = ''.join(mapping[torch.argmax(base).item()] for base in one_hot_sequence)
        return decoded_sequence
    elif one_hot_sequence.dim() == 3:  # Batch of sequences
        decoded_sequences = [''.join(mapping[torch.argmax(base).item()] for base in seq) for seq in one_hot_sequence]
        return decoded_sequences
    else:
        raise ValueError("Input tensor must be 2D or 3D.")

def reverse_complement(x):
    # Reverse the sequence batch-wise
    x = torch.flip(x, dims=[1])  # x is expected to be of shape (batch_size, seq_length, 4)
    x = torch.stack([x[:, :, 3], x[:, :, 2], x[:, :, 1], x[:, :, 0]], dim=-1)  # Complement the sequence (A<->T, C<->G)
    return x
