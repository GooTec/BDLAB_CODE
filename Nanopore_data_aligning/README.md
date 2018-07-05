# Nanopore data aligning

## Installation
### Minimap2
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

### Samtools
Building and Installing 
Building each desired package from source is very simple:
```sh
cd samtools-1.x    # and similarly for bcftools and htslib
./configure --prefix=/where/to/install
make
make install
```
See INSTALL in each of the source directories for further details.

The executable programs will be installed to a bin subdirectory under your specified prefix, so you may wish to add this directory to your $PATH:
```sh
export PATH=/where/to/install/bin:$PATH    # for sh or bash users
```

## Data Preparation
- reference.fasta 만들기
- Nanopore Sequencer를 통해 얻어진 reads중 pass 폴더에 있는 fastq 파일 이용

## Usage
### Aligning 
```sh
./minimap2 -ax map-ont home/tjahn/ref_new.fasta /home/tjahn/soap/data/pass/fastq_runid_9d57b64934d886aaade4ed350 b40a24e5bf3123b_1.fastq > run_1align.sam
```
reference(ref_new.fasta)와 nanopore를 통해 얻어진 query(reads)를 mapping 시켜서 .sam 형식의 output을 얻는다

.sam file에서 mapping quality와 CIGAR string 정보를 통해 어느정도 align되었는지 확인가능하다.
아래의 순서대로 진행하면 tview 라는 옵션을 통해 reference와 align된 정도를 눈으로 확인가능하다.
### Tview
```sh
samtools view -Sb ./run_1align.sam > ./run_1align.bam 
samtools sort ./run_1align.bam -o ./run1_align.bam.sorted
samtools index ./run1_align.bam.sorted
samtools tview ./run1_align.bam.sorted ref_new.fasta
```

### Pile up
```sh
samtools mpileup -f ./Desktop/samtools_output/ref_new.fasta
-s ./Desktop/output3/sorted/run1_align.bam.sorted
-o run1_pileup.txt #한 줄로 다 입력 !
```

- 시퀀스에서 각 자리마다 가장 많이 나온 서열이 무엇인지 voting 하는 옵션입니다.
- run1_pileup.txt 에서 각 줄의 3번째 column 들만 뽑아내면 reference와 같은 서열을 얻을 수 있습니다.
