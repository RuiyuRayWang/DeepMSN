import os
import yaml
import argparse
import numpy as np
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
    parser.add_argument('--sort', action='store_true', help='Sort the regions and topics before writing to bed file.')
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = config['dataset'] if 'dataset' in config else config

    num_topics = len(config['data_path'])
    genome_fasta = config['genome_fasta']
    region_to_topics = parse_bed_files(config['data_path'])
    regions_and_topics = list(region_to_topics.items())

    regions_and_topics_clean = clean_bed_entries(
        regions_and_topics = regions_and_topics, 
        genome_fasta = genome_fasta, 
        thread_pool_size=args.thread_pool_size
        )
        
    write_regions_and_topics_to_bed(
        regions_and_topics = regions_and_topics_clean,
        out_bed = os.path.join(config['out_dir'], 'regions_and_topics.bed')
        )

    if args.sort:
        print("Sorting the regions and topics in the BED file.")
        import pybedtools
        sorted_bed = pybedtools.BedTool(os.path.join(config['out_dir'], 'regions_and_topics.bed')).sort()
        sorted_bed.saveas(os.path.join(config['out_dir'], 'regions_and_topics_sorted.bed'))
        os.remove(os.path.join(config['out_dir'], 'regions_and_topics.bed'))
    
    print("Preprocessing completed successfully.")