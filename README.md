# VCF2PY – working with genomic variants in Python #

Package `vcf2py` allows to quickly import genomic
variants in VCF format into Python as NumPy arrays

## Installation ##

With pip 

    pip install vcf2py

## Sample usage ##

Import main class to work with VCF data:

    from vcf2py import VariantFile
    
### Parse variants and INFO fields ###

Consider the following VCF file `1KG_example.vcf`

    ##fileformat=VCFv4.3
    ##INFO=<ID=AF,Number=A,Type=Float,Description="Estimated allele frequency in the range (0,1)">
    ##INFO=<ID=AC,Number=A,Type=Integer,Description="Total number of alternate alleles in called genotypes">
    ##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of samples with data">
    ##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles in called genotypes">
    ##INFO=<ID=EAS_AF,Number=A,Type=Float,Description="Allele frequency in the EAS populations calculated from AC and AN, in the range (0,1)">
    ##INFO=<ID=EUR_AF,Number=A,Type=Float,Description="Allele frequency in the EUR populations calculated from AC and AN, in the range (0,1)">
    ##INFO=<ID=AFR_AF,Number=A,Type=Float,Description="Allele frequency in the AFR populations calculated from AC and AN, in the range (0,1)">
    ##INFO=<ID=AMR_AF,Number=A,Type=Float,Description="Allele frequency in the AMR populations calculated from AC and AN, in the range (0,1)">
    ##INFO=<ID=SAS_AF,Number=A,Type=Float,Description="Allele frequency in the SAS populations calculated from AC and AN, in the range (0,1)">
    ##INFO=<ID=VT,Number=.,Type=String,Description="indicates what type of variant the line represents">
    ##INFO=<ID=EX_TARGET,Number=0,Type=Flag,Description="indicates whether a variant is within the exon pull down target boundaries">
    ##INFO=<ID=DP,Number=1,Type=Integer,Description="Approximate read depth; some reads may have been filtered">
    #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00097	HG00099
    7	152135021	.	C	T	.	PASS	AC=0;AN=4;DP=21624;AF=0;EAS_AF=0;EUR_AF=0;AFR_AF=0;AMR_AF=0;SAS_AF=0;VT=SNP;NS=2548	GT	0|0	0|0
    7	152135047	.	T	C	.	PASS	AC=0;AN=4;DP=21003;AF=0;EAS_AF=0;EUR_AF=0;AFR_AF=0;AMR_AF=0;SAS_AF=0;VT=SNP;NS=2548	GT	0|0	0|0
    7	152135074	.	C	T	.	PASS	AC=0;AN=4;DP=20726;AF=0;EAS_AF=0;EUR_AF=0;AFR_AF=0;AMR_AF=0;SAS_AF=0.01;VT=SNP;NS=2548	GT	0|0	0|0
    7	152135149	.	A	G	.	PASS	AC=0;AN=4;DP=19360;AF=0;EAS_AF=0.01;EUR_AF=0;AFR_AF=0;AMR_AF=0;SAS_AF=0;VT=SNP;NS=2548	GT	0|0	0|0
    7	152135225	.	C	T	.	PASS	AC=0;AN=4;DP=20911;AF=0;EAS_AF=0;EUR_AF=0;AFR_AF=0;AMR_AF=0;SAS_AF=0;VT=SNP;NS=2548	GT	0|0	0|0
    7	152135289	.	A	G	.	PASS	AC=0;AN=4;DP=20973;AF=0;EAS_AF=0;EUR_AF=0.01;AFR_AF=0;AMR_AF=0;SAS_AF=0;VT=SNP;NS=2548	GT	0|0	0|0
    7	152135350	.	C	A	.	PASS	AC=0;AN=4;DP=20835;AF=0;EAS_AF=0;EUR_AF=0;AFR_AF=0;AMR_AF=0;SAS_AF=0;VT=SNP;NS=2548	GT	0|0	0|0

After parsing it gives the following

    >>> f = vcf.VariantFile("1KG_example.vcf")
    >>> vrt, info, samples = f.read(info=True)
    >>> info
    array([(0., 0, 2548, 4, 0.  , 0.  , 0., 0., 0.  , ('SNP',), False, 21624),
           (0., 0, 2548, 4, 0.  , 0.  , 0., 0., 0.  , ('SNP',), False, 21003),
           (0., 0, 2548, 4, 0.  , 0.  , 0., 0., 0.01, ('SNP',), False, 20726),
           (0., 0, 2548, 4, 0.01, 0.  , 0., 0., 0.  , ('SNP',), False, 19360),
           (0., 0, 2548, 4, 0.  , 0.  , 0., 0., 0.  , ('SNP',), False, 20911),
           (0., 0, 2548, 4, 0.  , 0.01, 0., 0., 0.  , ('SNP',), False, 20973),
           (0., 0, 2548, 4, 0.  , 0.  , 0., 0., 0.  , ('SNP',), False, 20835)],
          dtype=[('AF', '<f8'), ('AC', '<i8'), ('NS', '<i8'), ('AN', '<i8'), 
          ('EAS_AF', '<f8'), ('EUR_AF', '<f8'), ('AFR_AF', '<f8'), 
          ('AMR_AF', '<f8'), ('SAS_AF', '<f8'), ('VT', 'O'), ('EX_TARGET', '?'), ('DP', '<i8')])
          
Variants and their data can be easily imported into Pandas

    >>> import pandas as pd
    >>> pd.DataFrame.from_records(vrt)
      chrom        pos ref alt id  qual filter
    0     7  152135021   C   T  .   NaN   PASS
    1     7  152135047   T   C  .   NaN   PASS
    2     7  152135074   C   T  .   NaN   PASS
    3     7  152135149   A   G  .   NaN   PASS
    4     7  152135225   C   T  .   NaN   PASS
    5     7  152135289   A   G  .   NaN   PASS
    6     7  152135350   C   A  .   NaN   PASS

### Parsing genotypes ###

Genotypes in VCF files are declared as `String` type.  However they
are rather important in genetic analysis so additinal parsing tools
are implemented.  `genotypes` parameter of `VariantFile.read` provides
additional control.  In addition to default `genotypes="string"` it
allowes `split` or `sum` values.

Consider the following VCF file with three variants in total:

    ##fileformat=VCFv4.2
    ##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased Genotype">
    #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	SAMPLE1	SAMPLE2
    chr24	166	.	T	TG,C	100	.	.	GT	0/1/2	0/1/0
    chr24	167	.	T	TG	100	.	.	GT	./.	0/1

`genotypes=split` gives the following

    >>> from vcf2py import VariantFile
    >>> f = VariantFile("file.vcf")
    >>> vrt, info, samples = f.read(samples=True, genotypes="split")
    >>> samples["SAMPLE1"]["GT"]
    [[ 0  1  0]
     [ 0  0  1]
     [-1 -1 -2]]

i.e.  output will contain a ndarray with `1` or `0` depending on
presense of variant in allele for each of the variants.  `-1` stands
for `.` in GT field, `-2` is a filling value in case of smaller ploidy
to make array homogenous.


`genotypes=sum` gives

    >>> vrt, info, samples = f.read(samples=True, genotypes="sum")
    >>> samples["SAMPLE1"]["GT"]
    array([1, 1, 0], dtype=int8)

i.e. an integer value for each variant indicating number of alleles
with the variant.

