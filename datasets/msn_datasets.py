import torch
from torch.utils.data import Dataset
from Bio import SeqIO
import pybedtools
from utils.utils import one_hot_encode

class TestDataset(Dataset):
    # Input is a single bed file with 6 cols: chr, start, end, cell_type, -, -. 
    # The last two cols can be ignored.
    def __init__(self, fasta_file, bed_file, transform=None, target_transform=None):
        """
        Initialize the dataset with data and labels.
        :param fasta_file: Path to the FASTA file containing sequences.
        :param bed_file: Path to the BED file containing genomic regions.
        :param transform: Function to transform the sequence (e.g., one-hot encoding).
        :param target_transform: Function to transform the label.
        """
        self.fasta = fasta_file
        self.bed_entries = list(pybedtools.BedTool(bed_file))  # Convert BedTool to a list of entries
        self.transform = transform or one_hot_encode
        self.target_transform = target_transform or (lambda label: torch.tensor([1, 0]) if label == 'D1MSN' else torch.tensor([0, 1]))
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.bed_entries)  # Use the list of entries
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample and its label by index.
        :param idx: Index of the sample to retrieve.
        :return: A tuple (data, label) for the given index.
        """
        # Get fasta entry
        with open(self.fasta) as fasta_file:
            record_dict = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
        keys = list(record_dict)
        sequence = str(record_dict[keys[idx]].seq)
        
        # Get the bed entry and extract label
        bed_entry = self.bed_entries[idx]  # Use the list of entries
        label = bed_entry.name
        
        if self.transform:
            sequence = self.transform(sequence)
        if self.target_transform:
            label = self.target_transform(label)
        
        return sequence, label
