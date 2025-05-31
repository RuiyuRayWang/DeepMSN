import torch
from torch.utils.data import Dataset
import os
import pybedtools
from pyfaidx import Fasta
from utils.utils import one_hot_encode

class TopicDataset(Dataset):
    def __init__(self, config, augment=False, transform=None, target_transform=None):
    # def __init__(self, genome, region_topic_bed, num_topics=None, transform=None, target_transform=None):
        """
        Initialize the dataset with data and labels.
        :param config: Configuration dictionary containing dataset parameters
        :param transform: Function to transform the sequence (e.g., one-hot encoding).
        :param target_transform: Function to transform the label.
        """
        self.config = config.get('dataset', {})
        self.augment = augment
        self.augment_kwargs = self.config.get('augment_kwargs', {
            'extension_bp': 100,     # Extend region by 500bp on each side
            'window_size': 500,     # Size of sliding window
            'stride': 50,        # Number of random windows to sample
            'min_overlap': 0.5       # Minimum overlap with original region
        })
        
        # Get genome files from config
        self.genome_fasta = self.config.get('genome_fasta', None)
        self.chrom_sizes = self.config.get('chrom_sizes', None)
        
        # Load chromosome sizes for boundary checking
        self.chrom_sizes_dict = {}
        if self.chrom_sizes:
            with open(self.chrom_sizes, 'r') as f:
                for line in f:
                    chrom, size = line.strip().split('\t')
                    self.chrom_sizes_dict[chrom] = int(size)
        
        # Load bed entries from config-specified path
        regions_and_topics = os.path.join(self.config.get('out_dir', '.'), self.config.get('out_fn', 'regions_and_topics.bed'))
        self.bed_entries = list(pybedtools.BedTool(regions_and_topics))
        
        # Define the transform function
        self.transform = transform or one_hot_encode
        
        # Define the target transform function
        num_topics = len(self.config.get('data_path', [])) if self.config.get('data_path') else 0  # Get number of topics from config data_path
        if num_topics == 0:
            raise ValueError("Number of topics must be specified in the configuration.")
        self.target_transform = target_transform or (lambda label: torch.tensor([int(i in label) for i in map(str, range(num_topics))], dtype=torch.float))
    
        # If augmenting, pre-compute augmented regions for each original region
        self.augmented_regions = []
        if self.augment:
            self._precompute_augmented_regions()
    
    def _precompute_augmented_regions(self):
        """
        Pre-compute all possible augmented regions using sliding windows with stride.
        Each original bed entry will have multiple augmented coordinate variants.
        """
        extension_bp = self.augment_kwargs.get('extension_bp', 200)
        window_size = self.augment_kwargs.get('window_size', 800)
        stride = self.augment_kwargs.get('stride', 100)
        min_overlap = self.augment_kwargs.get('min_overlap', 0.5)
        
        for idx, bed_entry in enumerate(self.bed_entries):
            original_start = bed_entry.start
            original_end = bed_entry.end
            original_length = original_end - original_start
            chrom = bed_entry.chrom
            
            # Get chromosome size for boundary checking
            chrom_size = self.chrom_sizes_dict.get(chrom, float('inf'))
            
            # Extend the region
            extended_start = max(0, original_start - extension_bp)
            extended_end = min(chrom_size, original_end + extension_bp)
            
            # Always include the original region first
            self.augmented_regions.append((original_start, original_end, idx, 'original'))
            
            # Generate sliding windows with stride within the extended region
            window_start = extended_start
            
            while window_start + window_size <= extended_end:
                window_end = window_start + window_size
                
                # Calculate overlap with original region
                overlap_start = max(window_start, original_start)
                overlap_end = min(window_end, original_end)
                overlap_length = max(0, overlap_end - overlap_start)
                
                # Only include windows with sufficient overlap
                if overlap_length >= original_length * min_overlap:
                    # Skip if it's identical to the original region
                    if not (window_start == original_start and window_end == original_end):
                        aug_type = f'aug_{len([r for r in self.augmented_regions if r[2] == idx]) - 1}'
                        self.augmented_regions.append((window_start, window_end, idx, aug_type))
                
                # Move to next window position
                window_start += stride
        
        print(f"Generated {len(self.augmented_regions)} augmented regions from {len(self.bed_entries)} original regions")
        print(f"Augmentation factor: {len(self.augmented_regions) / len(self.bed_entries):.2f}x")
        
    def __len__(self):
        """
        Return the number of samples in the dataset (including augmented regions).
        """
        if self.augment:
            return len(self.augmented_regions)
        else:
            return len(self.bed_entries)
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample and its label by index.
        For augmented datasets, this returns one of the augmented coordinate variants.
        """
        # Open genome fasta
        fasta = Fasta(self.genome_fasta)
        
        if self.augment:
            # Get augmented coordinates
            start, end, original_idx, aug_type = self.augmented_regions[idx]
            bed_entry = self.bed_entries[original_idx]
            chrom = bed_entry.chrom
        else:
            # Get original coordinates
            bed_entry = self.bed_entries[idx]
            chrom = bed_entry.chrom
            start = bed_entry.start
            end = bed_entry.end
            original_idx = idx
            aug_type = 'original'
        
        # Extract the sequence from the fasta file using (potentially modified) coordinates
        seq = str(fasta[chrom][start:end]).upper()
        
        # Get the topic list from the original bed entry (labels don't change)
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
            'original_start': bed_entry.start,
            'original_end': bed_entry.end,
            'sequence': sequence, 
            'label': topic_vector,
            'index': original_idx,  # Keep track of original index
            'aug_type': aug_type
        }

    def get_augmentation_stats(self):
        """
        Get statistics about the augmentation process.
        """
        if not self.augment:
            return "No augmentation applied."
        
        original_count = len(self.bed_entries)
        augmented_count = len(self.augmented_regions)
        
        # Count augmentations per original region
        aug_per_region = {}
        for _, _, orig_idx, aug_type in self.augmented_regions:
            if orig_idx not in aug_per_region:
                aug_per_region[orig_idx] = []
            aug_per_region[orig_idx].append(aug_type)
        
        avg_aug_per_region = sum(len(augs) for augs in aug_per_region.values()) / len(aug_per_region)
        
        return {
            'original_regions': original_count,
            'total_augmented_regions': augmented_count,
            'augmentation_factor': augmented_count / original_count,
            'avg_augmentations_per_region': avg_aug_per_region,
            'min_augmentations': min(len(augs) for augs in aug_per_region.values()),
            'max_augmentations': max(len(augs) for augs in aug_per_region.values())
        }