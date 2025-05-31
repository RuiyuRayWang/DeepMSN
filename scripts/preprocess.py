import os
import yaml
import json
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pybedtools
from pyfaidx import Fasta
import random

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

def clean_bed_entries(regions_and_topics, genome_fasta):
    """
    Filter regions_and_topics to only include entries whose sequence (from genome_fasta) contains only valid bases (A, T, C, G).
    Returns a cleaned regions_and_topics list.
    """
    fasta = Fasta(genome_fasta)
    valid_bases = {'A', 'T', 'C', 'G'}
    cleaned = []
    
    print(f"Cleaning sequences for {len(regions_and_topics)} regions.")
    for (chrom, start, end), topic_list in tqdm(regions_and_topics, desc="Validating sequences"):
        seq = str(fasta[chrom][start:end]).upper()
        if set(seq).issubset(valid_bases):
            cleaned.append(((chrom, start, end), topic_list))
    
    fasta.close()
    return cleaned

def augment_regions_with_random_shift(regions_and_topics, genome_fasta, augment_kwargs, chrom_sizes_file=None):
    """
    Augment regions using random shift within extended boundaries.
    """
    extend = augment_kwargs.get('extension', 100)
    num_windows = augment_kwargs.get('num_windows', 5)
    
    # Load chromosome sizes
    chrom_sizes_dict = {}
    if chrom_sizes_file and os.path.exists(chrom_sizes_file):
        with open(chrom_sizes_file, 'r') as f:
            for line in f:
                chrom, size = line.strip().split('\t')
                chrom_sizes_dict[chrom] = int(size)
    
    augmented_regions = []
    fasta = Fasta(genome_fasta)
    valid_bases = {'A', 'T', 'C', 'G'}
    
    print(f"Augmenting {len(regions_and_topics)} regions with random shift...")
    print(f"Parameters: extend={extend}bp, num_windows={num_windows}")
    
    for region, topic_list in tqdm(regions_and_topics, desc="Random shift augmentation"):
        # Handle both formats: with and without strand
        if len(region) == 4:
            chrom, start, end, strand = region
        else:
            chrom, start, end = region
            strand = '+'  # Default to forward strand
        
        original_length = end - start
        region_augmented = []
        
        # Get chromosome size
        chrom_size = chrom_sizes_dict.get(chrom, float('inf'))
        
        # Calculate the extended region boundaries
        extended_start = max(0, start - extend)
        extended_end = min(chrom_size, end + extend)
        
        # Calculate valid shift range
        earliest_start = extended_start
        latest_start = extended_end - original_length
        latest_start = min(latest_start, chrom_size - original_length)
        
        # Always include the original region first
        region_augmented.append(((chrom, start, end, strand), topic_list))
        
        # Generate random shifted windows only if there's a valid range
        if latest_start > earliest_start:
            attempts = 0
            max_attempts = num_windows * 10
            
            while len(region_augmented) < num_windows and attempts < max_attempts:
                attempts += 1
                
                shift_start = random.randint(earliest_start, latest_start)
                shift_end = shift_start + original_length
                
                if shift_start == start:
                    continue
                    
                shifted_region = (chrom, shift_start, shift_end, strand)
                if any(existing[0] == shifted_region for existing in region_augmented):
                    continue
                
                try:
                    seq = str(fasta[chrom][shift_start:shift_end]).upper()
                    if set(seq).issubset(valid_bases):
                        region_augmented.append(((chrom, shift_start, shift_end, strand), topic_list))
                except Exception as e:
                    continue
        
        augmented_regions.extend(region_augmented)
    
    fasta.close()
    
    print(f"Generated {len(augmented_regions)} augmented regions from {len(regions_and_topics)} original regions")
    augmentation_factor = len(augmented_regions) / len(regions_and_topics) if len(regions_and_topics) > 0 else 1.0
    print(f"Augmentation factor: {augmentation_factor:.2f}x")
    
    return augmented_regions

def augment_regions_with_sliding_windows(regions_and_topics, genome_fasta, augment_kwargs, chrom_sizes_file=None):
    """
    Augment regions using sliding windows with coordinate extension.
    """
    extension_bp = augment_kwargs.get('extension', 100)
    window_size = augment_kwargs.get('window_size', 500)
    stride = augment_kwargs.get('stride', 50)
    
    # Load chromosome sizes
    chrom_sizes_dict = {}
    if chrom_sizes_file and os.path.exists(chrom_sizes_file):
        with open(chrom_sizes_file, 'r') as f:
            for line in f:
                chrom, size = line.strip().split('\t')
                chrom_sizes_dict[chrom] = int(size)
    
    augmented_regions = []
    fasta = Fasta(genome_fasta)
    valid_bases = {'A', 'T', 'C', 'G'}
    
    print(f"Augmenting {len(regions_and_topics)} regions with sliding windows...")
    print(f"Parameters: extension_bp={extension_bp}, window_size={window_size}, stride={stride}")
    
    for region, topic_list in tqdm(regions_and_topics, desc="Sliding window augmentation"):
        # Handle both formats: with and without strand
        if len(region) == 4:
            chrom, start, end, strand = region
        else:
            chrom, start, end = region
            strand = '+'  # Default to forward strand
        
        original_length = end - start
        region_augmented = []
        
        # Get chromosome size
        chrom_size = chrom_sizes_dict.get(chrom, float('inf'))
        
        # Extend the region
        extended_start = max(0, start - extension_bp)
        extended_end = min(chrom_size, end + extension_bp)
        
        # Always include the original region first
        region_augmented.append(((chrom, start, end, strand), topic_list))
        
        # Generate sliding windows with stride
        window_start = extended_start
        
        while window_start + window_size <= extended_end:
            window_end = window_start + window_size
            
            # Remove windows identical to original
            if not (window_start == start and window_end == end):
                try:
                    seq = str(fasta[chrom][window_start:window_end]).upper()
                    if set(seq).issubset(valid_bases):
                        region_augmented.append(((chrom, window_start, window_end, strand), topic_list))
                except Exception as e:
                    pass
            
            window_start += stride
        
        augmented_regions.extend(region_augmented)
    
    fasta.close()
    
    print(f"Generated {len(augmented_regions)} augmented regions from {len(regions_and_topics)} original regions")
    print(f"Augmentation factor: {len(augmented_regions) / len(regions_and_topics):.2f}x")
    
    return augmented_regions

def augment_regions_with_reverse_complement(regions_and_topics, genome_fasta):
    """
    Augment regions by adding reverse complement sequences as separate entries.
    This doubles the dataset size by creating new entries with strand information.
    """
    augmented_regions = []
    fasta = Fasta(genome_fasta)
    valid_bases = {'A', 'T', 'C', 'G'}
    
    print(f"Augmenting {len(regions_and_topics)} regions with reverse complement...")
    
    for region, topic_list in tqdm(regions_and_topics, desc="Reverse complement augmentation"):
        # Handle both formats: with and without strand
        if len(region) == 4:
            chrom, start, end, strand = region
        else:
            chrom, start, end = region
            strand = '+'  # Default to forward strand
        
        # Always include the original region (preserve existing strand)
        augmented_regions.append(((chrom, start, end, strand), topic_list))
        
        # Create reverse complement entry with opposite strand
        try:
            seq = str(fasta[chrom][start:end]).upper()
            if set(seq).issubset(valid_bases):
                opposite_strand = '-' if strand == '+' else '+'
                augmented_regions.append(((chrom, start, end, opposite_strand), topic_list))
        except Exception as e:
            pass
    
    fasta.close()
    
    print(f"Generated {len(augmented_regions)} regions from {len(regions_and_topics)} original regions (2x with reverse complement)")
    
    return augmented_regions

def split_regions_train_val_test(regions_and_topics, train_ratio=0.7, val_ratio=0.15, random_seed=42):
    """
    Split regions into train/validation/test sets.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Shuffle the regions
    regions_shuffled = regions_and_topics.copy()
    random.shuffle(regions_shuffled)
    
    total_regions = len(regions_shuffled)
    train_size = int(total_regions * train_ratio)
    val_size = int(total_regions * val_ratio)
    
    train_regions = regions_shuffled[:train_size]
    val_regions = regions_shuffled[train_size:train_size + val_size]
    test_regions = regions_shuffled[train_size + val_size:]
    
    print(f"Split {total_regions} regions into:")
    print(f"  Training: {len(train_regions)} ({len(train_regions)/total_regions:.1%})")
    print(f"  Validation: {len(val_regions)} ({len(val_regions)/total_regions:.1%})")
    print(f"  Test: {len(test_regions)} ({len(test_regions)/total_regions:.1%})")
    
    return train_regions, val_regions, test_regions

def write_regions_and_topics_to_bed(regions_and_topics, out_bed):
    """
    Write regions to BED format with strand information.
    """
    print(f"Writing {len(regions_and_topics)} regions and topics to {out_bed}.")
    with open(out_bed, 'w') as f:
        for region, topic_list in regions_and_topics:
            topic_str = ",".join(map(str, topic_list))
            
            # Handle both formats: with and without strand
            if len(region) == 4:  # Has strand information
                # Required for train set
                # Format: chrom, start, end, topic_str, ., strand
                chrom, start, end, strand = region
                f.write(f"{chrom}\t{start}\t{end}\t{topic_str}\t.\t{strand}\n")
            else:  # No strand information (original regions)
                # Required for val and test sets
                # Format: chrom, start, end, topic_str
                chrom, start, end = region
                f.write(f"{chrom}\t{start}\t{end}\t{topic_str}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess genomic data with train/val/test split and augmentation.")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file.')
    # parser.add_argument('-t', '--thread_pool_size', type=int, default=8, help='Number of threads for parallel processing.')
    args = parser.parse_args()
    
    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    tmp_dir = config.get('tmp_dir', '/tmp')
    
    config_ds = config.get('dataset', {})
    if not config_ds:
        raise ValueError("No dataset configuration found in the provided config file.")
    
    out_dir = config_ds.get('out_dir', './output')
    out_fn = config_ds.get('out_fn', 'regions_and_topics.bed')
    
    # Get parameters from config
    train_ratio = config_ds.get('train_ratio', 0.7)
    val_ratio = config_ds.get('val_ratio', 0.15)
    random_seed = config_ds.get('random_seed', 42)
    
    augment = config_ds.get('augment', False)
    augment_kwargs = config_ds.get('augment_kwargs', {})
    
    num_topics = len(config_ds['data_path'])
    genome_fasta = config_ds['genome_fasta']
    chrom_sizes = config_ds.get('chrom_sizes', None)
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    region_to_topics = parse_bed_files(config_ds['data_path'])
    regions_and_topics = list(region_to_topics.items())
    
    # Clean invalid sequences
    regions_and_topics_clean = clean_bed_entries(
        regions_and_topics=regions_and_topics, 
        genome_fasta=genome_fasta
    )
    
    # Split into train/val/test BEFORE augmentation
    train_regions, val_regions, test_regions = split_regions_train_val_test(
        regions_and_topics_clean,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed
    )
    
    # Apply augmentation only to training data if requested
    if augment:
        print("Augmenting training data...")
        
        # Choose augmentation method based on config
        random_shift = augment_kwargs.get('random_shift', False)
        reverse_complement = config_ds.get('rc', False)
        
        if random_shift:
            print("Using random shift augmentation...")
            train_regions_augmented = augment_regions_with_random_shift(
                train_regions, 
                genome_fasta, 
                augment_kwargs,
                chrom_sizes
            )
        else:
            print("Using sliding window augmentation...")
            train_regions_augmented = augment_regions_with_sliding_windows(
                train_regions, 
                genome_fasta, 
                augment_kwargs,
                chrom_sizes
            )
            
        # Apply reverse complement augmentation if requested
        if reverse_complement:
            print("Applying reverse complement augmentation...")
            train_regions_augmented = augment_regions_with_reverse_complement(
                train_regions_augmented, genome_fasta
            )
    else:
        train_regions_augmented = train_regions
        
        # Apply reverse complement even without other augmentation if requested
        if config_ds.get('rc', False):
            print("Applying reverse complement augmentation...")
            train_regions_augmented = augment_regions_with_reverse_complement(
                train_regions_augmented, genome_fasta
            )
        else:
            print("No augmentation applied to training data.")
    
    # Create output filenames
    train_fn = out_fn.replace('.bed', '_train.bed')
    val_fn = out_fn.replace('.bed', '_val.bed')
    test_fn = out_fn.replace('.bed', '_test.bed')
    
    # Write train/val/test sets to separate BED files
    datasets = [
        (regions_and_topics, out_fn, "original"),
        (train_regions_augmented, train_fn, "training"),
        (val_regions, val_fn, "validation"), 
        (test_regions, test_fn, "test")
    ]
    
    for regions, filename, dataset_name in datasets:
        temp_file = os.path.join(tmp_dir, filename)
        final_file = os.path.join(out_dir, filename)
        
        # Write to temp file
        write_regions_and_topics_to_bed(regions, temp_file)
        
        # Sort and save to final location
        print(f"Sorting {dataset_name} regions...")
        sorted_bed = pybedtools.BedTool(temp_file).sort()
        sorted_bed.saveas(final_file)
        
        # Clean up temp file
        os.remove(temp_file)
        
        print(f"{dataset_name.capitalize()} set saved to: {final_file}")
    
    # Save split indices for reproducibility
    indices_file = os.path.join(out_dir, 'split_info.json')
    split_info = {
        "preprocessing_info": {
            "random_seed": random_seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": 1.0 - train_ratio - val_ratio,
            "augmentation": augment,
            "augmentation_method": "random_shift" if augment and augment_kwargs.get('random_shift', False) else "sliding_window" if augment else None,
            # "thread_pool_size": args.thread_pool_size,
            "config_file": os.path.abspath(config_path),
            "timestamp": __import__('datetime').datetime.now().isoformat()
        },
        "augmentation_params": augment_kwargs if augment else None,
        "dataset_sizes": {
            "original_total": len(regions_and_topics_clean),
            "train_original": len(train_regions),
            "train_after_augmentation": len(train_regions_augmented),
            "validation": len(val_regions),
            "test": len(test_regions)
        },
        "file_paths": {
            "train_file": train_fn,
            "val_file": val_fn,
            "test_file": test_fn,
            "genome_fasta": genome_fasta,
            "chrom_sizes": chrom_sizes
        },
        "augmentation_stats": {
            "augmentation_factor": len(train_regions_augmented) / len(train_regions) if len(train_regions) > 0 else 1.0,
            "regions_added": len(train_regions_augmented) - len(train_regions)
        } if augment else None
    }

    with open(indices_file, 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"Split information saved to: {indices_file}")
    print("Preprocessing completed successfully.")