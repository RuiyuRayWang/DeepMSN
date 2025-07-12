import os
import torch
import numpy as np

from Bio.Seq import Seq, MutableSeq

def one_hot_encode(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [1/4, 1/4, 1/4, 1/4],
               'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [1/4, 1/4, 1/4, 1/4]}
    one_hot_encoded = [mapping[base] for seq in sequence for base in seq]
    return torch.tensor(one_hot_encoded, dtype=torch.float32)

# def decode_one_hot(one_hot_sequence):
#     mapping = ['A', 'C', 'G', 'T']
#     if one_hot_sequence.dim() == 2:  # Single sequence
#         decoded_sequence = ''.join(mapping[torch.argmax(base).item()] for base in one_hot_sequence)
#         return decoded_sequence
#     elif one_hot_sequence.dim() == 3:  # Batch of sequences
#         decoded_sequences = [''.join(mapping[torch.argmax(base).item()] for base in seq) for seq in one_hot_sequence]
#         return decoded_sequences
#     else:
#         raise ValueError("Input tensor must be 2D or 3D.")

def reverse_complement(x):
    # Reverse the sequence batch-wise
    x = torch.flip(x, dims=[1])  # x is expected to be of shape (batch_size, seq_length, 4)
    x = torch.stack([x[:, :, 3], x[:, :, 2], x[:, :, 1], x[:, :, 0]], dim=-1)  # Complement the sequence (A<->T, C<->G)
    return x

def generate_seq(seq_length=500, gc_content=0.4, immutable=False):
    """
    Generates a random DNA sequence with a specified GC content.
    """
    gc_count = int(seq_length * gc_content)
    at_count = seq_length - gc_count
    sequence = np.random.choice(['G', 'C'], gc_count).tolist() + np.random.choice(['A', 'T'], at_count).tolist()
    np.random.shuffle(sequence)
    seq_str = ''.join(sequence)
    return Seq(seq_str) if immutable else MutableSeq(seq_str)

def saturated_mutagenesis(sequence, positions=None):
    """
    Given a single sequence, generate all possible single point mutations and outputs a list.
    Input sequence should be a MutableSeq object.
    
    Args:
        sequence (MutableSeq): The input sequence to mutate.
        positions (list, optional): A list of positions (0-based index) to mutate. 
                                    If None, all positions will be mutated.
    
    Returns:
        list: A list of mutated sequences.
    """
    nucleotides = ['A', 'T', 'C', 'G']
    seqs = []
    
    # If positions are not specified, mutate all positions
    if positions is None:
        positions = range(len(sequence))
    
    for i in positions:
        original_nucleotide = sequence[i]
        for nucleotide in nucleotides:
            if nucleotide != original_nucleotide:
                # Create a copy of the sequence to avoid modifying the original
                new_seq = sequence[:]
                new_seq[i] = nucleotide
                seqs.append(new_seq)
    
    return seqs

def load_model_from_safetensors(checkpoint_path, config_path, model_class, set_eval=True):
    """
    Load model from Accelerate-saved SafeTensors format
    Adapted from evaluation notebook in DeepMSN repository.
    """
    
    import yaml
    from safetensors.torch import load_file
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = model_class(config=config)
    
    # Load from SafeTensors
    safetensors_path = os.path.join(checkpoint_path, 'model.safetensors')
    
    if os.path.exists(safetensors_path):
        # Load the state dict from SafeTensors
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {safetensors_path}")
    else:
        raise FileNotFoundError(f"SafeTensors file not found at {safetensors_path}")
    
    # Set to eval mode and move to device
    if set_eval:
        model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    return model, config