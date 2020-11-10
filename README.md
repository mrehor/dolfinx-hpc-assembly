# dolfinx-hpc-assembly

Tests for comparison of parallel efficiency of assembly routines in
FEniCS/DOLFINX (<https://www.fenicsproject.org/>).

## Weak scaling test

The list of options for the weak scaling test can be obtained by running
```bash
python3 test_assembly_routines.py -h
```

*Remark*: pandas (<https://pandas.pydata.org/>) has to be installed, e.g. with
`python3 -m pip install pandas`, in order to store the results in a CSV file that can be further
postprocessed with
```bash
./output_assembly_routines.py <filename>
```
