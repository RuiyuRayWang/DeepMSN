import argparse
from Bio import SeqIO

def parse_fasta(input, output, invalid_regions=None):
    """
    Parses a FASTA file to remove sequences containing non-standard nucleotides.
    Enforces nucleotide letters to be in uppercase.
    Records and reports the number of entries processed and dropped.
    Optionally writes headers of dropped entries to a BED file.
    :param input: Path to the input FASTA file.
    :param output: Path to the output FASTA file.
    :param invalid_regions: Path to the BED file for storing headers of dropped entries.
    """
    total_entries = 0
    dropped_entries = 0

    with open(output, "w") as output_handle, \
         open(invalid_regions, "w") if invalid_regions else open("/dev/null", "w") as invalid_handle:
        for record in SeqIO.parse(input, "fasta"):
            total_entries += 1
            sequence = record.seq.upper()  # Must convert to uppercase
            if set(sequence).issubset({"A", "T", "C", "G"}):
                record.seq = sequence
                SeqIO.write(record, output_handle, "fasta")
            else:
                dropped_entries += 1
                if invalid_regions:
                    # Extract chrom:start-end from the header
                    chrom, coords = record.id.split(":")
                    start, end = coords.split("-")
                    invalid_handle.write(f"{chrom}\t{start}\t{end}\n")

    print(f"Total entries processed: {total_entries}")
    print(f"Entries dropped: {dropped_entries}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a FASTA file to remove sequences with non-standard nucleotides.")
    parser.add_argument("input", type=str, help="Path to the input FASTA file.")
    parser.add_argument("output", type=str, help="Path to the output FASTA file.")
    parser.add_argument("-bed", "--invalid-regions", type=str, help="Path to the BED file for dropped entries.")
    args = parser.parse_args()

    parse_fasta(args.input, args.output, invalid_regions=args.invalid_regions)
