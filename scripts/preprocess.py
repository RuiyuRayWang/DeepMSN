import os
import yaml
import argparse
import numpy as np
from utils.utils import one_hot_encode
# from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pyfaidx import Fasta

def parse_bed_files(bed_files):
    region_to_topics = {}
    for bed_file, label in bed_files:
        with open(bed_file, 'r') as bed:
            for line in bed:
                chrom, start, end = line.strip().split()[:3]
                start, end = int(start) - 1, int(end)
                region_key = (chrom, start, end)
                if region_key not in region_to_topics:
                    region_to_topics[region_key] = []
                region_to_topics[region_key].append(label)
    print(f"Parsed {len(region_to_topics)} regions from bed files.")
    return region_to_topics

def augment_regions(regions_and_topics, extension=0, window_size=None, stride=None, chrom_size=None):
    augmented = []
    for (chrom, start, end), topic_list in regions_and_topics:
        # Extend
        ext_start = max(0, start - extension) if extension else start
        ext_end = end + extension if extension else end
        # Sliding window
        if window_size and stride:
            for win_start in range(ext_start, ext_end - window_size + 1, stride):
                win_end = win_start + window_size
                # Check chromosome boundary
                if chrom_size is not None and chrom in chrom_size:
                    if win_end > chrom_size[chrom]:
                        continue  # skip this window
                augmented.append(((chrom, win_start, win_end), topic_list))
        else:
            # Check chromosome boundary
            if chrom_size is not None and chrom in chrom_size:
                if ext_end > chrom_size[chrom]:
                    continue  # skip this region
            augmented.append(((chrom, ext_start, ext_end), topic_list))
    return augmented

# def parse_seq_label(
#     regions_and_topics,
#     genome_fasta,
#     num_topics
#     ):
#     """
#     Parallelly extract sequences and their corresponding labels from the genome FASTA file.
#     Filter out sequences with invalid bases (non-ATCG).
#     """
#     fasta = Fasta(genome_fasta)
#     valid_bases = {'A', 'T', 'C', 'G'}
#     sequences = []
#     labels = []
#     n = 0
    
#     def fetch_seq_label(args):
#         (chrom, start, end), topic_list = args
#         seq = str(fasta[chrom][start:end]).upper()
#         if set(seq).issubset(valid_bases):
#             topic_vector = [0] * num_topics
#             for topic in topic_list:
#                 topic_vector[topic] = 1
#             return seq, topic_vector
#         else:
#             return None
    
#     print(f"Extracting sequences for {len(regions_and_topics)} regions.")
#     with ThreadPoolExecutor(max_workers=8) as executor:
#         for result in tqdm(executor.map(fetch_seq_label, regions_and_topics), total=len(regions_and_topics), desc="Extracting sequences"):
#             if result is not None:
#                 seq, topic_vector = result
#                 sequences.append(seq)
#                 labels.append(topic_vector)
#                 n += 1
#     fasta.close()
    
#     # print(f"One-hot encoding {len(sequences)} sequences.")
#     # X = one_hot_encode(sequences)
#     y = np.array(labels)
#     return sequences, y, n

def clean_bed_entries(regions_and_topics, genome_fasta, thread_pool_size=8):
    """
    Filter regions_and_topics to only include entries whose sequence (from genome_fasta) contains only valid bases (A, T, C, G).
    Returns a cleaned regions_and_topics list.
    """
    fasta = Fasta(genome_fasta)
    valid_bases = {'A', 'T', 'C', 'G'}
    cleaned = []

    def is_valid(args):
        (chrom, start, end), topic_list = args
        seq = str(fasta[chrom][start:end]).upper()
        return set(seq).issubset(valid_bases)

    print(f"Cleaning sequences for {len(regions_and_topics)} regions.")
    with ThreadPoolExecutor(max_workers=thread_pool_size) as executor:
        results = list(tqdm(executor.map(is_valid, regions_and_topics), total=len(regions_and_topics), desc="Validating sequences"))
        for i, valid in enumerate(results):
            if valid:
                cleaned.append(regions_and_topics[i])
    fasta.close()
    return cleaned

def write_regions_and_topics_to_bed(regions_and_topics, out_bed):
    print(f"Writing {len(regions_and_topics)} regions and topics to {out_bed}.")
    with open(out_bed, 'w') as f:
        for (chrom, start, end), topic_list in regions_and_topics:
            topic_str = ",".join(map(str, topic_list))
            f.write(f"{chrom}\t{start}\t{end}\t{topic_str}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess genomic data.")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('-t', '--thread_pool_size', type=int, default=8, help='Number of threads for parallel processing.')
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = config['dataset'] if 'dataset' in config else config

    num_topics = len(config['data_path'])
    genome_fasta = config['genome_fasta']
    region_to_topics = parse_bed_files(config['data_path'])
    regions_and_topics = list(region_to_topics.items())

    # Train/val/test split (on regions, before augmentation)
    np.random.seed(42)
    np.random.shuffle(regions_and_topics)
    N = len(regions_and_topics)

    regions_and_topics_clean = clean_bed_entries(
        regions_and_topics = regions_and_topics, 
        genome_fasta = genome_fasta, 
        thread_pool_size=args.thread_pool_size
        )
    
    write_regions_and_topics_to_bed(
        regions_and_topics = regions_and_topics_clean,
        out_bed = os.path.join(config['out_dir'], 'regions_and_topics.bed')
        )
    
    # # Load chromosome sizes if available
    # chrom_size = None
    # if 'chrom_sizes' in config:
    #     chrom_size = {}
    #     with open(config['chrom_sizes']) as f:
    #         for line in f:
    #             chrom, size = line.strip().split()[:2]
    #             chrom_size[chrom] = int(size)

    # Extract and encode for each split
    # X, y, n = parse_seq_label(regions_and_topics, genome_fasta, num_topics)

    # Save each split
    # np.savez(os.path.join(config['out_dir'],'data.npz'), X=X, y=y)
    # print(f"Done. Saved {X.shape[0]}/{n} samples.")