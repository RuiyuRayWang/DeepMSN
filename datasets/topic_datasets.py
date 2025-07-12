import torch
from torch.utils.data import Dataset
import os
import random
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
        
        self.augment = self.config.get('augment', False)
        self.augment_kwargs = self.config.get('augment_kwargs', {})
        
        self.augment_multiplier = self.augment_kwargs.get('augment_multiplier', 3) if self.augment else 1
    
    def __len__(self):
        """
        Return the number of samples in the dataset including augmentations.
        """
        return len(self.bed_entries) * self.augment_multiplier
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample and its label by index.
        """
        # Map augmented index back to original sample
        original_idx = idx % len(self.bed_entries)
        augment_variant = idx // len(self.bed_entries)
        
        # Open genome fasta
        fasta = Fasta(self.genome_fasta)
        
        # Get the bed entry
        bed_entry = self.bed_entries[original_idx]
        chrom = bed_entry.chrom
        original_start = bed_entry.start
        original_end = bed_entry.end
        original_length = original_end - original_start
        
        # Always apply random shift augmentation if enabled (no original copies)
        if self.augment:
            start, end = self.apply_random_shift(original_start, original_end, chrom, fasta)
        else:
            start, end = original_start, original_end
        
        # Extract sequence
        seq = str(fasta[chrom][start:end]).upper()
        
        # # Always apply reverse complement augmentation probabilistically if enabled
        is_rc = False
        if self.augment:
            rc_prob = self.augment_kwargs.get('rc_prob', 0.0)
            if random.random() < rc_prob:
                seq = self.reverse_complement(seq)
                is_rc = True
        
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
            'original_start': original_start,
            'original_end': original_end,
            'sequence': sequence, 
            'label': topic_vector,
            'index': idx,
            'original_index': original_idx,
            'augment_variant': augment_variant,
            'is_rc': is_rc
        }
    
    def apply_random_shift(self, original_start, original_end, chrom, fasta):
        """
        Apply random shift augmentation by extending the region and randomly shifting within it.
        Takes chromosome boundaries into account to prevent exceeding chromosome limits.
        """
        original_length = original_end - original_start
        extension = self.augment_kwargs.get('extension', 50)
        
        # Get chromosome size - use fasta length if chrom_sizes not available
        chrom_sizes_file = self.config.get('chrom_sizes', None)
        chrom_size = len(fasta[chrom])  # Default to fasta chromosome length
        
        # Load chromosome sizes from file if available
        if chrom_sizes_file and os.path.exists(chrom_sizes_file):
            try:
                with open(chrom_sizes_file, 'r') as f:
                    for line in f:
                        chrom_name, size = line.strip().split('\t')
                        if chrom_name == chrom:
                            chrom_size = int(size)
                            break
            except Exception:
                # Fall back to fasta length if file reading fails
                pass
        
        # Create extended region with chromosome boundaries
        extended_start = max(0, original_start - extension)
        extended_end = min(chrom_size, original_end + extension)
        
        # Calculate valid shift range
        earliest_start = extended_start
        latest_start = extended_end - original_length
        latest_start = min(latest_start, chrom_size - original_length)
        
        # Check if shifting is possible
        if latest_start <= earliest_start:
            # No room for shifting, return original coordinates
            return original_start, original_end
        
        # Random shift within the valid range
        shift_start = random.randint(earliest_start, latest_start)
        shift_end = shift_start + original_length
        
        # Ensure we don't exceed chromosome boundary
        if shift_end > chrom_size:
            shift_end = chrom_size
            shift_start = shift_end - original_length
        
        return shift_start, shift_end

    def reverse_complement(self, seq):
        """Return reverse complement of DNA sequence"""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return ''.join(complement.get(base, 'N') for base in seq[::-1])