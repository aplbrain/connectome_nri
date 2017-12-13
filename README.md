# connectome_nri

This repo contains MATLAB and Python functions that can be used to calculate NRI metrics (NRI, precision, and recall) given a count table of matching synaptic terminals.

Demo scripts are provided for each language.


The count table in the demo files is:

```
   [0,   100,    15,    10,   200]
   [10,    1,    10,   300,    20]
   [5,    10,   100,     5,    10]
```

For both demo files, the output for the count table above should be:

```
=========================
Results from nri()
=========================
nriNet =  0.642756410256
precNet =  0.559261531597
recallNet =  0.75555723005

=========================
Results from nri_slow()
=========================
nriNet =  0.642756410256
precNet =  0.559261531597
recallNet =  0.75555723005
```
