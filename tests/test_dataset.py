from datasets.topic_datasets import TopicDataset
from torch.utils.data import DataLoader

dataset = TopicDataset(
    genome='data/resources/mm10.fa',
    region_topic_bed='data/CTdnsmpl_catlas_35_Topics_top_3k/regions_and_topics_sorted.bed',
    transform=None,  # Use default one-hot encoding
    target_transform=None  # Use default target transformation
)

dataset[0]

