import torch
from torch.utils.data import Dataset
import pybedtools
from pyfaidx import Fasta
from utils.utils import one_hot_encode

class TopicDataset(Dataset):
    # Input is a dictionary of bed files with labels.
    def __init__(self, genome, region_topic_bed, transform=None, target_transform=None):
        """
        Initialize the dataset with data and labels.
        :param genome: Path to the genome FASTA file.
        :param region_topic_bed: Path to the BED file containing genomic regions and their associated topics.
        :param transform: Function to transform the sequence (e.g., one-hot encoding).
        :param target_transform: Function to transform the label.
        """
        self.genome_fasta = genome
        self.bed_entries = list(pybedtools.BedTool(region_topic_bed))  # Convert BedTool to a list of entries
        self.transform = transform or one_hot_encode
        num_topics = 18
        self.target_transform = target_transform or (lambda label: torch.tensor([int(i in label) for i in map(str, range(num_topics))], dtype=torch.int))
    
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
        
        # Extract the sequence from the fasta file
        seq = str(fasta[chrom][start:end]).upper()
        
        # Get the topic list from the bed entry
        label = bed_entry.name.split(',')
        
        # One hot encode the sequence
        sequence = self.transform(seq)
        
        # Get topic vector
        topic_vector = self.target_transform(label)
        
        fasta.close()
        
        return {'sequence': sequence, 'label': topic_vector}
