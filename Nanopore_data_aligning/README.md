# Nanopore data aligning

## Installation
```sh
git clone https://github.com/lh3/minimap2
cd minimap2 && make
# long sequences against a reference genome
./minimap2 -a test/MT-human.fa test/MT-orang.fa > test.sam
# create an index first and then map
./minimap2 -d MT-human.mmi test/MT-human.fa
./minimap2 -a MT-human.mmi test/MT-orang.fa > test.sam
# use presets (no test data)
./minimap2 -ax map-pb ref.fa pacbio.fq.gz > aln.sam       # PacBio genomic reads
./minimap2 -ax map-ont ref.fa ont.fq.gz > aln.sam         # Oxford Nanopore genomic reads
./minimap2 -ax sr ref.fa read1.fa read2.fa > aln.sam      # short genomic paired-end reads
./minimap2 -ax splice ref.fa rna-reads.fa > aln.sam       # spliced long reads
./minimap2 -ax splice -k14 -uf ref.fa reads.fa > aln.sam  # Nanopore Direct RNA-seq
./minimap2 -cx asm5 asm1.fa asm2.fa > aln.paf             # intra-species asm-to-asm alignment
./minimap2 -x ava-pb reads.fa reads.fa > overlaps.paf     # PacBio read overlap
./minimap2 -x ava-ont reads.fa reads.fa > overlaps.paf    # Nanopore read overlap
# man page for detailed command line options
man ./minimap2.1
```

