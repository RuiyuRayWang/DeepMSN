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
        self.augment_kwargs = config.get('augment_kwargs', {})
        
        # Get path to genome fasta from config
        self.genome_fasta = self.config.get('genome_fasta', None)
        
        # Load bed entries from config-specified path
        bed_file_path = os.path.join(self.config.('out_dir', '.'), self.config.get('out_fn', 'regions_and_topics.bed'))
        self.bed_entries = list(pybedtools.BedTool(bed_file_path))
        
        self.transform = transform or one_hot_encode
        
        # Get number of topics from config data_path
        num_topics = len(self.config.get('data_path', [])) if self.config.get('data_path') else 0
        if num_topics == 0:
            raise ValueError("Number of topics must be specified in the configuration.")
        self.target_transform = target_transform or (lambda label: torch.tensor([int(i in label) for i in map(str, range(num_topics))], dtype=torch.float))
    
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
        # Open genome fasta
        fasta = Fasta(self.genome_fasta)
        
        # Get the bed entry
        bed_entry = self.bed_entries[idx]
        chrom = bed_entry.chrom
        start = bed_entry.start
        end = bed_entry.end
        
        if self.augment:
            
        # Extract the sequence from the fasta file
        seq = str(fasta[chrom][start:end]).upper()
        
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
            'index': idx
        }
