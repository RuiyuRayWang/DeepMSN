import torch
from torch.utils.data import Dataset
import os
import pybedtools
from pyfaidx import Fasta
from utils.utils import one_hot_encode

class TopicDataset(Dataset):
    def __init__(self, config, transform=None, target_transform=None):
        """
        Initialize the dataset with data and labels.
        :param config: Configuration dictionary containing dataset parameters
        :param transform: Function to transform the sequence (e.g., one-hot encoding).
        :param target_transform: Function to transform the label.
        """
        self.config = config.get('dataset', {})
        
        # Get path to genome fasta from config
        self.genome_fasta = self.config.get('genome_fasta', None)
        
        # Load bed entries from bed file
        bed_file_path = self.config['bed_file']
        if not os.path.exists(bed_file_path):
            raise FileNotFoundError(f"BED file not found: {bed_file_path}")
        self.bed_entries = list(pybedtools.BedTool(bed_file_path))
        
        # Define the transformation functions
        self.transform = transform or one_hot_encode
        
        # Define the target transformation function
        num_topics = len(self.config.get('data_path', [])) if self.config.get('data_path') else 0
        if num_topics == 0:
            raise ValueError("Failed to infer number of topics from the configuration file.")
        self.target_transform = target_transform or (lambda label: torch.tensor([int(i in label) for i in map(str, range(num_topics))], dtype=torch.float))
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.bed_entries)
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample and its label by index.
        """
        # Open genome fasta
        fasta = Fasta(self.genome_fasta)
        
        # Get the bed entry
        bed_entry = self.bed_entries[idx]
        chrom = bed_entry.chrom
        start = bed_entry.start
        end = bed_entry.end
        
        # Check for strand information (6th column)
        strand = '+'  # Default to forward
        if hasattr(bed_entry, 'strand') and bed_entry.strand:
            strand = bed_entry.strand
        elif len(bed_entry.fields) >= 6:
            strand = bed_entry.fields[5]
        
        # Extract sequence
        seq = str(fasta[chrom][start:end]).upper()
        
        # Apply reverse complement if strand is negative
        if strand == '-':
            seq = self.reverse_complement(seq)
        
        # Get the topic list from the bed entry
        label = bed_entry.name.split(',')
        
        # One hot encode the sequence
        sequence = self.transform(seq)
        
        # Get topic vector
        topic_vector = self.target_transform(label)
        
        fasta.close()
        
        return {
            'chrom': chrom,
            'start': start,
            'end': end,
            'sequence': sequence, 
            'label': topic_vector,
            'index': idx,
            'strand': strand,
            'is_rc': strand == '-'
        }

    def reverse_complement(self, seq):
        """Return reverse complement of DNA sequence"""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return ''.join(complement.get(base, 'N') for base in seq[::-1])