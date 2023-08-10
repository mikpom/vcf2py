# Time-stamp: <2023-08-09 16:45:17>
import os
import sys
from importlib import resources
import unittest
import numpy as np
import pandas as pd
from vcf2py import VariantFile
import tempfile

VCF_HEADER = [
    """##fileformat=VCFv4.2""",
    """#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO""",
]
GT_SPEC = (
    """##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased Genotype">"""
)
AC_SPEC = """##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count in genotypes, for each ALT allele, in the same order as listed">"""
NUMBER2INTFIELD_SPEC = (
    """##INFO=<ID=FieldNum2,Number=2,Type=Integer,Description="Desc">"""
)
NUMBERDOTINTFIELD_SPEC = (
    """##INFO=<ID=FieldNumDot,Number=.,Type=Integer,Description="Desc">"""
)


class VCFFileTestCaseBase(unittest.TestCase):
    def setUp(self, extra_info=None, extra_gt=None, samples=None):
        self.testdir = tempfile.TemporaryDirectory()
        self.testfile = os.path.join(self.testdir.name, type(self).__name__)
        with open(self.testfile, "wt") as fh:
            # Write first line
            fh.write(VCF_HEADER[0])
            fh.write("\n")

            # Write Extra fields definitions if provided
            if extra_info is not None:
                for infoline in extra_info:
                    fh.write(infoline)
                    fh.write("\n")
            if extra_gt is not None:
                for fmtline in extra_gt:
                    fh.write(fmtline)
                    fh.write("\n")
            # Write last line of header
            if samples is not None:
                fh.write(
                    VCF_HEADER[-1] + "\t" + "\t".join(["FORMAT"] + samples)
                )
                fh.write("\n")
            else:
                fh.write(VCF_HEADER[-1])
                fh.write("\n")

    def tearDown(self):
        self.testdir.cleanup()


class VCFFileTestCase2Samples(VCFFileTestCaseBase):
    def setUp(self):
        super().setUp(extra_gt=[GT_SPEC], samples=["SAMPLE1", "SAMPLE2"])


class UndeclaredFormatField(VCFFileTestCase2Samples):
    def setUp(self):
        super().setUp()
        with open(self.testfile, "at") as fh:
            fh.write("chr24\t166\t.\tT\tTG\t100\t.\t.\tGT:TEST\t1/1:100\t0/0")

    def test_parsing(self):
        f = VariantFile(self.testfile)
        self.assertEqual(f.samples, ["SAMPLE1", "SAMPLE2"])
        vrt, info, samples = f.read(info=True, samples=True)
        self.assertEqual(vrt.shape, (1,))
        gt = samples[0]["SAMPLE1"]["GT"]
        self.assertEqual(gt, "1/1")


class ParsingGenotypes(VCFFileTestCase2Samples):
    def setUp(self):
        super().setUp()
        with open(self.testfile, "at") as fh:
            fh.write("chr24\t166\t.\tT\tTG\t100\t.\t.\tGT	0|0	0|1\n")
            fh.write("chr24\t167\t.\tT\tTG\t100\t.\t.\tGT	0|0	0|1\n")

    def test_genotypes_as_string(self):
        f = VariantFile(self.testfile)
        vrt, info, samples = f.read(
            info=True,
            samples=True,
            genotypes="string",
        )

        self.assertEqual(samples.shape, (2,))  # 2 variants
        # Check genotypes of the first variant
        self.assertEqual(samples["SAMPLE2"][0]["GT"], "0|1")
        self.assertEqual(samples["SAMPLE1"][0]["GT"], "0|0")

    def test_genotypes_split(self):
        f = VariantFile(self.testfile)
        # Now the same but with genotypes as arrays
        vrt, info, samples = f.read(
            info=True,
            samples=True,
            genotypes="split",
        )
        SAMPLE2gt = samples["SAMPLE2"]["GT"]
        self.assertEqual(SAMPLE2gt.shape, (2, 2))  # 2 variants x diploid
        self.assertEqual(SAMPLE2gt[0][0], 0)
        self.assertEqual(SAMPLE2gt[0][1], 1)

    def test_genotypes_sum(self):
        f = VariantFile(self.testfile)
        # Now the same but with genotypes as sum of alleles
        vrt, info, samples = f.read(
            info=True,
            samples=True,
            genotypes="sum",
        )
        SAMPLE2gt = samples["SAMPLE2"]["GT"]
        self.assertEqual(SAMPLE2gt.shape, (2,))  # 2 variants x 2 samples
        self.assertEqual(SAMPLE2gt[0], 1)


class ParseVLGenotypes(VCFFileTestCase2Samples):
    """Tests on example1.vcf file focusing on variable
    GT sizes and missing values"""

    def setUp(self):
        super().setUp()
        with open(self.testfile, "at") as fh:
            fh.write("chr24\t166\t.\tT\tTG,C\t100\t.\t.\tGT	0/1/2	0/1/0\n")
            fh.write("chr24\t167\t.\tT\tTG\t100\t.\t.\tGT	./.	0/1\n")

    def test_genotypes_split(self):
        f = VariantFile(self.testfile)
        vrt, info, samples = f.read(info=True, samples=True, genotypes="split")

        gt = samples["SAMPLE1"]["GT"]

        self.assertEqual(gt[0][0], 0)
        self.assertEqual(gt[0][1], 1)

        self.assertEqual(gt[2][0], -1)  # ./. First dot
        self.assertEqual(gt[2][1], -1)  # ./. Second dot
        self.assertEqual(gt[2][2], -2)  # Nothing but present due to GT above

    def test_genotypes_sum(self):
        f = VariantFile(self.testfile)
        vrt, info, samples = f.read(info=True, samples=True, genotypes="sum")

        gt = samples["SAMPLE1"]["GT"]

        self.assertEqual(gt[0], 1)
        self.assertEqual(gt[0], 1)

        self.assertEqual(gt[2], 0)  # Zero number of 3-rd variant @pos 167


class ParseMissingValues(VCFFileTestCaseBase):
    def setUp(self):
        PL_SPEC = """##INFO=<ID=pl,Number=1,Type=Integer,Description="Description">"""
        AD_SPEC = """##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depth">"""
        super().setUp(
            extra_info=[PL_SPEC],
            extra_gt=[GT_SPEC, AD_SPEC],
            samples=["SAMPLE1", "SAMPLE2"],
        )
        with open(self.testfile, "at") as fh:
            fh.write(
                "chr24\t166\t.\tT\tTG\t100\t.\tpl=.\tGT:AD	0|0:.	0|1:./.\n"
            )
            fh.write(
                "chr24\t167\t.\tT\tTG\t100\t.\t.\tGT:AD	0|0:10,10	0|1:20,20\n"
            )

    def test_missing(self):
        f = VariantFile(self.testfile)
        vrt, info, samples = f.read(info=True, samples=True)

        # Test missing INFO field
        pl = info["pl"]
        self.assertTrue(np.isnan(pl[0]))
        self.assertTrue(np.isnan(pl[1]))

        ad = samples["SAMPLE1"]["AD"]
        self.assertTrue(np.isnan(ad[0]))  # First variant
        self.assertEqual(ad[1], 10)  # Second variant


class ParseVCFSeveralChunks(VCFFileTestCase2Samples):
    """Case reproduces a VCF with size spanning several chunks to read"""

    def setUp(self):
        super().setUp()
        with open(self.testfile, "at") as fh:
            for i in range(VariantFile.READ_CHUNK + 10):
                fh.write(f"chr24\t{1+i}\t.\tT\tC\t100\t.\t.\tGT	0/1	0/1")
                fh.write("\n")

            fh.write(
                f"chr24\t{VariantFile.READ_CHUNK + 100}\t.\tT\tC\t100\t.\t.\tGT	1/1/1	0/1"
            )
            fh.write("\n")

    def test_genotypes_split(self):
        f = VariantFile(self.testfile)
        vrt, info, samples = f.read(info=True, samples=True, genotypes="split")

        gt = samples["SAMPLE1"]["GT"]
        # Last variant with GT 1/1/1
        self.assertEqual(gt[-1][0], 1)
        self.assertEqual(gt[-1][1], 1)
        self.assertEqual(gt[-1][2], 1)

    def test_genotypes_sum(self):
        f = VariantFile(self.testfile)
        vrt, info, samples = f.read(info=True, samples=True, genotypes="sum")

        gt = samples["SAMPLE1"]["GT"]
        self.assertEqual(gt[-1], 3)


class ACField(VCFFileTestCaseBase):
    def setUp(self):
        super().setUp(extra_info=[AC_SPEC])
        with open(self.testfile, "at") as fh:
            fh.write("chr1	57856	.	T	A	403.06	PASS	AC=2\n")

    def test_AC_field(self):
        f = VariantFile(self.testfile)
        vrt, info, samples = f.read(info=True)
        self.assertEqual(info["AC"][0], 2)


class IntFieldNumber2(VCFFileTestCaseBase):
    def setUp(self):
        super().setUp(extra_info=[NUMBER2INTFIELD_SPEC])
        with open(self.testfile, "at") as fh:
            fh.write("chr1	57856	.	T	A	403.06	PASS	.\n")
            fh.write("chr1	57857	.	T	A	403.06	PASS	FieldNum2=10,10\n")

    def test_field(self):
        f = VariantFile(self.testfile)
        vrt, info, samples = f.read(info=True)

        self.assertEqual(info["FieldNum2"][0].shape, (2,))
        self.assertTrue(all([np.isnan(v) for v in info["FieldNum2"][0]]))

        self.assertEqual(tuple(info["FieldNum2"][1]), (10, 10))


class IntFieldNumberUnknown(VCFFileTestCaseBase):
    def setUp(self):
        super().setUp(extra_info=[NUMBERDOTINTFIELD_SPEC])
        with open(self.testfile, "at") as fh:
            fh.write("chr1	57856	.	T	A	403.06	PASS	.\n")
            fh.write("chr1	57857	.	T	A	403.06	PASS	FieldNumDot=10,10\n")

    def test_field(self):
        f = VariantFile(self.testfile)
        vrt, info, samples = f.read(info=True)

        self.assertEqual(type(info["FieldNumDot"][0]), tuple)
        self.assertEqual(info["FieldNumDot"][0][0], None)

        self.assertEqual(tuple(info["FieldNumDot"][1]), (10, 10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
