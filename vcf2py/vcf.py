"""Module contains classes needed for VCF parsing.

Class :class:`VCFReader`  which is instantiated from VCF file. 
VCF can be gzipped. Bgzipping and tabix-derived indexing is also 
supported for random coordinate-based access.

"""

import os
import sys
import warnings
from functools import lru_cache
from itertools import (
    dropwhile,
    groupby,
    repeat,
    takewhile,
    zip_longest,
    islice,
    chain,
    repeat,
)
from collections import namedtuple, OrderedDict, deque, defaultdict
import pathlib
import re
import gzip
import itertools
import io
import numpy as np

GT_VALUE_UNKNOWN = -1  # a dot  in GT, for example ./.
GT_VALUE_UNSPECIFIED = -2  # for cases of unhomogenous GT arrays

# Dtype for storing variant information in NumPy Array
DTYPE0 = [
    ("chrom", object),
    ("pos", int),
    ("ref", object),
    ("alt", object),
    ("id", object),
    ("qual", np.float_),
    ("filter", object),
]


# Map of VCF types to NumPy types
def grouper(it, n, fill=None):
    """returns empty iterator if `it` is empty"""
    args = [it] * n
    return itertools.zip_longest(*args, fillvalue=fill)

vcf_numpy_types = {
    "Float": np.float64,
    "Integer": np.int64,
    "String": object,
    "Flag": bool,
    "Character": "U1",
}
vcf_python_types = {
    "Float": np.float64,
    "Integer": np.int64,
    "String": object,
    "Flag": bool,
    "Character": str,
}

VCF_FIELDS = [
    "CHROM",
    "POS",
    "ID",
    "REF",
    "ALT",
    "QUAL",
    "FILTER",
    "INFO",
    "FORMAT",
    "SAMPLES",
]


class VCFRow(object):
    """Object storing variant information in VCF-like form."""

    MANDATORY_FIELDS = VCF_FIELDS[:8]

    def __init__(
        self,
        CHROM,
        POS,
        ID,
        REF,
        ALT,
        QUAL,
        FILTER,
        INFO,
        FORMAT=None,
        SAMPLES=None,
        rnum=None,
    ):
        self.CHROM = CHROM
        self.POS = int(POS)
        self.ID = ID
        self.REF = REF
        self.ALT = ALT
        self.QUAL = QUAL
        self.FILTER = FILTER
        self.INFO = INFO
        self.FORMAT = FORMAT
        self.SAMPLES = SAMPLES
        self.rnum = rnum

    __slots__ = [*VCF_FIELDS, "rnum"]

    @staticmethod
    def _to_string(v):
        if v is None:
            return "."
        else:
            return str(v)

    def __repr__(self):
        return "<VCFRow {}:{} {}->{}>".format(
            self.CHROM, self.POS, self.REF, self.ALT
        )

    def __str__(self):
        if not self.FORMAT is None:
            return "\t".join(
                self._to_string(getattr(self, a)) for a in VCF_FIELDS
            )
        else:
            return "\t".join(
                self._to_string(getattr(self, a))
                for a in self.MANDATORY_FIELDS
            )


class VariantFile(object):
    """Class to read VCF files."""

    def __init__(self, vcf, index=None):
        """
        VariantFile is instantiated from a VCF file. Optionally,
        index argument can be given.

        Parameters
        ----------
        vcf : path-like
            Path to file with variants
        
        index : str or bool
            By default tries ``vcf``.tbi as index file.
            if index=True and index not found NoIndexFoundError
            is raised.
            A string with path to index can be given. *Default: None*

        Returns
        -------
        reader : VariantFile
            VariantFile object.
        """
        if isinstance(vcf, io.TextIOBase):
            self.buf = vcf
            self.opened_file = False
        else:  # path-like object
            self.fl = pathlib.PosixPath(vcf)
            if self.fl.suffix in (".gz", ".bgz"):
                self.compressed = True
            else:
                self.compressed = False
            self.idx_file = self.check_index(self.fl, index)
            if self.compressed:
                self.openfn = gzip.open
            else:
                self.openfn = open
            self.buf = self.openfn(self.fl, "rt")
            self.opened_file = True

        self._vrt_start = False
        self.info_spec = OrderedDict()
        self.format_spec = OrderedDict()

        self.samples = []
        for cnt, line in enumerate(self.buf):
            if line.startswith("##INFO"):
                dat = self._parse_dtype(line)
                self.info_spec[dat["name"]] = dat
            elif line.startswith("##FORMAT"):
                dat = self._parse_dtype(line)
                self.format_spec[dat["name"]] = dat
            elif line.startswith("#CHROM"):
                # this should be the last header line
                self.header_len = cnt + 1

                vals = line.strip().split("\t")
                if len(vals) > 9:
                    self.samples = vals[9:]
                self._vrt_start = True
            elif not line.startswith("#"):
                break

        if self.opened_file:
            self.buf.close()
        self.sample_ind = OrderedDict()
        for ind, sample in enumerate(self.samples):
            self.sample_ind[sample] = ind

        self.default_parsers = {}
        self.default_converters = {}
        self.default_splitters = {
            "info": defaultdict(lambda: identity_func),
            "format": defaultdict(lambda: identity_func),
        }
        self.missing_values = {}
        for info_or_format, spec in zip(
            ["info", "format"], [self.info_spec, self.format_spec]
        ):
            for field, props in spec.items():
                tp = props["python_type"]
                sz = props["numpy_size"]
                num = props["number"]
                # if info_or_format == "format" and field == "GT":
                #     self.parsers[info_or_format][field] = _parse_gt
                # else:
                if sz>1:
                    miss_value = [None] * sz
                elif num in ['G', '.']:
                    miss_value = (None,)
                else:
                    miss_value = None
                self.missing_values.setdefault(info_or_format, {}).setdefault(field, miss_value)
                self.default_parsers.setdefault(info_or_format, {}).setdefault(
                    field, self._make_parser_func(tp, num)
                )
                if num in [0, 1]:
                    self.default_splitters[info_or_format][field] = lambda v: v
                else:
                    self.default_splitters[info_or_format][
                        field
                    ] = lambda v: v.split(",")
                self.default_converters.setdefault(
                    info_or_format, {}
                ).setdefault(field, VariantFile._get_converter(props))

    @staticmethod
    @lru_cache(maxsize=100)
    def _parse_gt(gt, ind):
        """Function return genotype tuple given genotype string and allele index.
        For performance results are cached based on arguments"""
        vals = re.split("[/|]", gt)
        GT = []
        for val in vals:
            if val == ".":
                GT.append(GT_VALUE_UNKNOWN)
            elif int(val) == ind + 1:
                GT.append(1)
            else:
                GT.append(0)
        GT = tuple(GT)
        return GT

    @staticmethod
    def _isindexed(file):
        idx = file + ".tbi"
        if os.path.isfile(idx):
            if os.path.getmtime(idx) >= os.path.getmtime(file):
                return True
            else:
                warnings.warn("Index is outdated for {}".format(file))
        return False

    @classmethod
    def _make_parser_func(cls, python_type, num, convert=True):
        """Returns a parser function based on TYPE and NUMBER arguments."""
        if convert == False:
            conv = lambda v: v
        else:
            if python_type == "U1":
                conv = str
            elif python_type == object:
                conv = lambda v: v
            else:
                conv = python_type
        if num == "R":

            def f(v, n):
                try:
                    return conv(v[n + 1])
                except IndexError:
                    return None
                except ValueError as exc:
                    if v[n + 1] == ".":
                        return None
                    else:
                        raise exc

        elif num == "A":

            def f(v, n):
                try:
                    return conv(v[n])
                except IndexError:
                    return None
                except ValueError as exc:
                    if v[n] == ".":
                        return None
                    else:
                        raise exc

        elif num == 1:
            if issubclass(python_type, bool):
                f = lambda v, n: True
            else:

                def f(v, n):
                    try:
                        return conv(v)
                    except ValueError as exc:
                        if v == ".":
                            return None
                        else:
                            raise exc

        elif num == 0:
            if issubclass(python_type, bool):
                f = lambda v, n: True
            else:
                f = lambda v, n: None
        elif num == "G" or num == ".":
            f = lambda v, n: tuple(map(conv, v))
        else:  # num 2,3 ...

            def f(v, n):
                try:
                    return tuple([conv(v[i]) for i in range(num)])
                except IndexError as exc:
                    r = []
                    for i in range(num):
                        if i > len(v) - 1:
                            r.append(None)
                        else:
                            r.append(conv(v[i]))
                    return tuple(r)

        return f

    @staticmethod
    def check_index(file, index):
        if index is True or index is None:
            idx = file.with_suffix(file.suffix + ".tbi")
            if idx.exists:
                return idx
            elif index is True:
                raise NoIndexFoundError("no index found for " + str(file))
            else:
                return None
        else:
            if os.path.isfile(index):
                return index
            else:
                raise ValueError("{} not found".format(index))

    @classmethod
    def _get_numpy_dtype(cls, spec, int_as_float):
        """Return a suitable NumPy datatype for storage based on field
        specification parsed from VCF"""
        # Integers are stored as floats because of NaNs
        if spec["number"] in ["G", "."]:  # As variable-length tuples
            return object
        elif spec["type"] == "Integer" and int_as_float:
            basetype = np.float64
        else:
            basetype = spec["numpy_type"]
        if spec["numpy_size"] == 1:
            return basetype
        else:
            return (basetype, spec["numpy_size"])

    @classmethod
    def _get_converter(cls, spec):
        dtype = cls._get_numpy_dtype(spec, int_as_float=True)

        def converter(values):
            ar = np.zeros(dtype=dtype, shape=len(values))
            ar[:] = values
            return ar

        return converter

    @staticmethod
    def _parse_dtype(line):
        """Parses VCF header and returns correct datatypes."""
        rx = (
            "##(?:(?:INFO)|(?:FORMAT))=\<ID=(\S+),Number=([\w.]+),"
            + 'Type=(\w+),Description="(.*)"(:?,Source="(.*)")?(?:,Version=".*")?\>'
        )
        match = re.match(rx, line)
        NAME, NUMBER, TYPE, DESCRIPTION, SOURCE, VERSION = match.groups()

        try:
            num = int(NUMBER)
        except ValueError:
            num = NUMBER
        if isinstance(num, int):
            if num == 0:
                size = 1
                numpy_type = bool
                python_type = bool
            else:
                size = num
                numpy_type = vcf_numpy_types[TYPE]
                python_type = vcf_python_types[TYPE]
        elif num in ("A", "R"):
            size = 1
            numpy_type = vcf_numpy_types[TYPE]
            python_type = vcf_python_types[TYPE]
        elif num in (".", "G"):
            # in numpy stored as links to python tuples
            size = 1
            numpy_type = object
            # Every item in a tuple is converted accordingly
            python_type = vcf_python_types[TYPE]
            
        else:
            raise ValueError("Unknown NUMBER:" + num)
        return {
            "name": NAME,
            "type": TYPE,
            "numpy_type": numpy_type,
            "numpy_size": size,
            "python_type" : python_type,
            "number": num,
            "description": DESCRIPTION,
        }

    def iter_rows(self, check_order=False):
        """Yields rows of variant file"""

        buf = self.get_buf()
        buf.seek(0)
        return RowIterator(self, buf, check_order)

    # Number of rows read per iteration in `VariantFile.read` function
    READ_CHUNK = 1000

    def read(self, info=False, samples=False, region=None, genotypes="string"):
        """
        Returns parsed variant data as a dict of NumPy arrays with structured
        dtype.
        
        Parameters
        ----------
        info : bool
            If True INFO fields a parsed.  *Default: False*
        samples : bool
            If True sample data columnes are parsed *Default: False*
        region : tuple(str, int, int)
            Genomic region to read data. Region is specified
            as a tuple (chromosome, start end).

            If not given, the whole file is parsed.

        genotypes : str (string|split|sum)
            How to parse GT fields *Default: string*

            By default GT field is parsed according to spec, i.e.
            a string e.g. of form "1/2".

            If `split` then for each variant in the output
            will contain a ndarray with 1 or 0 depending
            on presense of allele in particular sample.
            -1 stands for `.` in GT field.

            If `sum` a integer with the number of alleles
            containing variant is given.

        Returns
        -------
        variants, info, samples : three ndarrays
            `info` and `samples` can be None depending on
            the value of corresponding parameters.

        Examples
        --------

        Parsing genotypes.

        Consider the following VCF file
        
        ##fileformat=VCFv4.2
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased Genotype">
        #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	SAMPLE1	SAMPLE2
        chr24	166	.	T	TG,C	100	.	.	GT	0/1/2	0/1/0
        chr24	167	.	T	TG	100	.	.	GT	./.	0/1

        >>> from vcf2py import VariantFile
        >>> f = VariantFile("file.vcf")
        >>> vrt, info, samples = f.read(samples=True, genotypes="split")
        >>> vrt
        array([('chr24', 166, 'T', 'TG', '.', 100., '.'),
               ('chr24', 166, 'T', 'C', '.', 100., '.'),
               ('chr24', 167, 'T', 'TG', '.', 100., '.')],
                dtype=[('chrom', 'O'), ('pos', '<i8'), ('ref', 'O'),
                       ('alt', 'O'), ('id', 'O'), ('qual', '<f8'),
                       ('filter', 'O')])
        >>> samples["SAMPLE1"]["GT"]
        [[ 0  1  0]
         [ 0  0  1]
         [-1 -1 -2]]

        """
        if region is not None:
            rows = self._query(*region)
        else:
            rows = self.iter_rows()

        parse_info = info
        parse_samples = samples

        dataparser = self.get_parser(genotypes)
        # For chunk in chunks:
        #   read data as tuples with fields (Part 1)
        #   convert tuples to arrays (Part 2)
        # concatenate arrays into strunctured NumPy array (Part 3)

        # Field chunks
        fchunks = {
            "vrt": {f: [] for f, dtype in DTYPE0},
            "info": {f: [] for f in self.info_spec},
            "sampdata": {
                s: {f: [] for f in self.format_spec} for s in self.samples
            },
        }
        vnum = 0

        # rows are chunked. Then every chunk of rows gives a chunk of
        # records correspoding to fields of a structured array
        for row_chunk in grouper(rows, self.READ_CHUNK):
            # Will never reach here if rows is empty
            # Part 1
            tups = {
                "vrt": [],
                "info": [],
                # 'vcf':[],
                "sampdata": {s: [] for s in self.samples},
            }
            for row in row_chunk:
                if row is None:  # reached last row
                    break
                alts = row.ALT.split(",")
                if parse_info:  # Read INFO if necessary
                    orderedinfo = dataparser.get_info(row.INFO, alts=len(alts))
                else:
                    orderedinfo = repeat(None)
                if parse_samples and row.SAMPLES:
                    sampdata = dataparser.get_sampdata(
                        row, len(alts), genotypes
                    )
                else:
                    sampdata = repeat(None)
                for _alt, _info, _sampdata in zip(alts, orderedinfo, sampdata):
                    # Adding VCF notation related fields
                    tups["vrt"].append(
                        [
                            row.CHROM,
                            row.POS,
                            row.REF,
                            _alt,
                            row.ID,
                            row.QUAL if row.QUAL != "." else None,
                            row.FILTER,
                        ]
                    )
                    if parse_info:
                        tups["info"].append(_info)
                    if parse_samples:
                        for sn in range(len(self.samples)):
                            tups["sampdata"][self.samples[sn]].append(
                                _sampdata[sn]
                            )
            # Part 1 ends here

            # Part 2 (Tuples are zipped into arrays)
            zipped = {
                k: list(zip_longest(*tups[k], fillvalue=0))
                for k in ("vrt", "info")
            }
            zipped["sampdata"] = {}
            for samp in self.samples:
                zipped["sampdata"][samp] = list(zip(*tups["sampdata"][samp]))
            cur_sz = len(tups["vrt"])
            for ind, (field, dtype) in enumerate(DTYPE0):
                a = np.zeros(dtype=dtype, shape=cur_sz)
                a[:cur_sz] = zipped["vrt"][ind]
                fchunks["vrt"][field].append(np.resize(a, cur_sz))

            if parse_info:
                for ind, (field, spec) in enumerate(self.info_spec.items()):
                    converter_func = dataparser.converters["info"][field]
                    fchunks["info"][field].append(
                        converter_func(zipped["info"][ind])
                    )

            if parse_samples:
                for samp in self.samples:
                    for ind, (field, spec) in enumerate(
                        self.format_spec.items()
                    ):
                        converter_func = dataparser.converters["format"][field]
                        fchunks["sampdata"][samp][field].append(
                            converter_func(zipped["sampdata"][samp][ind])
                        )
            vnum += cur_sz
            # Part 2 ends here

        # Part 3
        vrt = np.zeros(dtype=DTYPE0, shape=vnum)
        for field, dtype in DTYPE0:
            vrt[field][:] = np.concatenate(fchunks["vrt"][field])
        # Part 3 info
        if parse_info:
            # default_dtype = self.dataparser.get_dtype('info')
            concat = {}
            dtype = []
            for ind, (field, vals) in enumerate(fchunks["info"].items()):
                # field_name, field_dtype, *rest = default_dtype[ind]
                spec = self.info_spec[field]
                a, rdtype = self._concat_chunks(spec, vals)
                dtype.append(rdtype)
                concat[field] = a
            info = np.zeros(dtype=dtype, shape=vnum)
            for field, array in concat.items():
                info[field][:] = array
        else:
            info = None
        # Part 3 samples
        if parse_samples:
            concat = {}
            dtype = {}  # Dtype as will be in output of function call
            for sampind, samp in enumerate(self.samples):
                for ind, (field, vals) in enumerate(
                    fchunks["sampdata"][samp].items()
                ):
                    if field == "GT":
                        concat_func = self._concat_chunks_gt
                        concat_args = vals, genotypes
                    else:
                        concat_func = self._concat_chunks
                        concat_args = [self.format_spec[field], vals]

                    ret, field_dtype = concat_func(*concat_args)
                    concat.setdefault(samp, {}).setdefault(field, ret)
                    dtype.setdefault(samp, []).append(field_dtype)

            sampdata = np.zeros(shape=(vnum,), dtype=list(dtype.items()))
            for samp in self.samples:
                for field in self.format_spec:
                    sampdata[samp][field] = concat[samp][field]
        else:
            sampdata = None
        # Part 3 ends here

        return vrt, info, sampdata

    def get_parser(self, genotypes):
        parsers = {k: v.copy() for k, v in self.default_parsers.items()}
        converters = {k: v.copy() for k, v in self.default_converters.items()}

        if genotypes != "string":
            parsers["format"]["GT"] = self._parse_gt
            converters["format"]["GT"] = self.gt_to_numpy

        return _DataParser(
            self.info_spec,
            self.format_spec,
            parsers,
            converters,
            self.default_splitters,
            self.missing_values,
            len(self.samples),
        )

    def _concat_chunks(self, spec, vals):
        """Concatenates values and return a dtype how the values are
        stored"""
        a = np.concatenate(vals)
        if (
            spec["type"] == "Integer" and spec["numpy_type"] != object
        ):  # Coerce to floats
            # or preserve ints
            nans = np.isnan(a).sum()
            if nans > 0:  # Storing as floats due to NaNs
                dtype = (
                    spec["name"],
                    self._get_numpy_dtype(spec, int_as_float=True),
                )
                ret = a
            else:
                ret = a.astype(int)

                dtype = (
                    spec["name"],
                    self._get_numpy_dtype(spec, int_as_float=False),
                )
        else:
            ret = a
            dtype = (
                spec["name"],
                self._get_numpy_dtype(spec, int_as_float=False),
            )

        return ret, dtype

    @staticmethod
    def gt_to_numpy(values):
        # The following trick fills short genotypes
        # with a missing value so array becomes
        # homogenous
        # TODO unnecessary if already homogenous
        gt_homogen = list(
            zip(
                *itertools.zip_longest(*values, fillvalue=GT_VALUE_UNSPECIFIED)
            )
        )
        ar = np.zeros(
            shape=(len(gt_homogen), len(gt_homogen[0])), dtype=np.int8
        )
        ar[:] = gt_homogen
        return ar

    def _concat_chunks_gt(self, values, genotypes):
        if genotypes == "split":
            min_gt_len = min([v.shape[1] for v in values])
            max_gt_len = max([v.shape[1] for v in values])
            # Genotypes might be of different length in the chunk
            # First they need to be padded to the same shape
            if max_gt_len!=min_gt_len:
                padded = [
                    np.pad(
                        v,
                        pad_width=[(0, 0), (0, max_gt_len - v.shape[1])],
                        constant_values=GT_VALUE_UNSPECIFIED,
                    )
                    for v in values
                ]
            else:
                padded = values
            gt = np.concatenate(padded)
            ret = np.asarray(gt)
            dtype = ("GT", np.int8, (gt.shape[1],))

        elif genotypes == "sum":
            gtfixed = [np.where(v < 0, 0, v) for v in values]
            gtsum = [np.sum(v, axis=1) for v in gtfixed]
            ret = np.concatenate(gtsum)
            dtype = ("GT", np.int8)
        elif genotypes == "string":
            ret = np.concatenate(values)
            dtype = ("GT", object)
        return ret, dtype

    def _get_tabix(self):
        if hasattr(self, "tabix"):
            return self.tabix
        else:
            if self.idx_file:
                try:
                    self.tabix = pysam.TabixFile(
                        filename=self.fl, index=self.idx_file
                    )
                except NameError:
                    import pysam

                    pysam.set_verbosity(0)
                    self.tabix = pysam.TabixFile(
                        filename=self.fl, index=self.idx_file
                    )
                except OSError as exc:
                    msg = exc.args[0]
                    if "index" in msg and "not found" in msg:
                        raise NoIndexFoundError
                    else:
                        raise exc
            else:
                raise NoIndexFoundError("No idx file")
        return self.tabix

    def _query(self, chrom, start=None, end=None):
        """User tabix index to fetch VCFRow's"""
        if start and not end:
            raise ValueError('"start" was given but "end" was not')
        tabix = self._get_tabix()
        try:
            lines = tabix.fetch(chrom, start, end)
        except ValueError:  # pysam raises on wrong chromosome
            return
        for line in lines:
            yield VCFRow(*line.strip().split("\t", maxsplit=9))

    def get_chroms(self):
        """Returns ``ChromSet`` corresponding to VCF. If indexed
        then index is used for faster access. Alternatively if ``allow_no_index``
        is True the whole file is parsed to get chromosome ordering."""
        try:
            tabix = self._get_tabix()
            self._chroms = tabix.contigs
            return self._chroms
        except NoIndexFoundError:
            raise ValueError("Index required to get chromosomes")

    def get_buf(self):
        if self.opened_file:
            r = self.openfn(self.fl, "rt")
            return r
        else:
            return self.buf

    def close(self):
        if self.opened_file:
            self.buf.close()

    @property
    def chroms(self):
        if hasattr(self, "_chroms"):
            return self._chroms
        else:
            self.get_chroms()
            return self._chroms

    @chroms.deleter
    def chroms(self):
        delattr(self, "_chroms")


class _DataParser(object):
    """Object for parsing INFO and SAMPLES data."""

    def __init__(
        self,
        info_spec,
        format_spec,
        parsers,
        converters,
        splitters,
        missing_values,
        nsamples,
    ):
        self.info_spec = info_spec
        self.format_spec = format_spec
        self.parsers = parsers
        self.converters = converters
        self.splitters = splitters
        self.missing_values = missing_values

        self.order = {
            "info": {f: ind for ind, f in enumerate(info_spec)},
            "format": {f: ind for ind, f in enumerate(format_spec)},
        }
        self.nsamples = nsamples

    def tokenize_info(self, INFO):
        """Function splits info on simple key value pairs"""
        info = []
        for field in INFO.split(";"):
            if not field or field == ".":
                continue
            keyval = field.split("=", maxsplit=1)
            if len(keyval) == 1:
                info.append((field, None))
            else:
                key, val = keyval
                info.append((key, self.splitters["info"][key](val)))
        return info

    def tokenize_sampdata(self, FORMAT, SAMPLES):
        fmt = FORMAT.split(":")
        _samples = SAMPLES.split("\t")
        splitters = self.splitters["format"]
        sampdata = [
            [(k, splitters[k](v)) for k, v in zip(fmt, d.split(":"))]
            for d in _samples
        ]
        return sampdata

    def get_info(self, INFO, alts=1, parse_null=False):
        """Given INFO string and number of alt alleles returns a list
        of lists with data corresponding to alleles, then fields."""
        tokenized = self.tokenize_info(INFO)
        info2 = []
        for an in range(alts):
            info1 = [
                self.missing_values["info"][field]
                for i, field in enumerate(self.info_spec)
            ]
            for ind, (key, val) in enumerate(tokenized):
                try:
                    conv = self.parsers["info"][key]
                except KeyError:
                    conv = no_parser

                try:
                    v = conv(val, an)
                except ValueError as exc:
                    if val == "" or val == ".":
                        continue # missing value not touched
                    else:
                        raise exc
                info1[self.order["info"][key]] = v
            info2.append(info1)

        return info2

    def get_sampdata(self, row, alts, genotypes):
        """Given SAMPLE data string and number of alt alleles returns a list
        of lists of lists with data corresponding to alleles, then samples,
        then fields."""
        # Return values is a list by field by sample by alt allele
        sampdata = [[None for j in range(self.nsamples)] for i in range(alts)]

        # _samples is a list of lists by field by sample
        # values are already split according to declared Number
        _samples = self.tokenize_sampdata(row.FORMAT, row.SAMPLES)

        for an in range(alts):
            for sn in range(self.nsamples):
                # _sampdata is a list by fields
                _sampdata = [
                    self.missing_values["format"][field]
                    for i, field in enumerate(self.format_spec)
                ]
                for key, val in _samples[sn]:
                    try:
                        order = self.order["format"][key]
                    except KeyError:
                        # Format field is not declared in the header
                        continue
                    # if key == "GT" and genotypes == "string":
                    #     parser = no_parser
                    # else:
                    #     parser =
                    v = self.parsers["format"][key](val, an)
                    _sampdata[order] = v
                sampdata[an][sn] = _sampdata
        return sampdata


class RowIterator:
    def iterate(self):
        cnt = 0
        for line in dropwhile(lambda l: l.startswith("#"), self.fh):
            row = VCFRow(*line.strip().split("\t", maxsplit=9), rnum=cnt)
            yield row
            cnt += 1
        self.close()

    def __init__(self, reader, fh, check_order):
        self.fh = fh
        self.check_order = check_order
        self.reader = reader
        if self.check_order:
            self.iterator = _check_VCF_order(self.iterate())
        else:
            self.iterator = self.iterate()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator)

    def close(self):
        self.fh.close()


no_parser = lambda v, a: v # VariantFile._make_parser_func(np.object_, 1, convert=False)
identity_func = lambda v: v
