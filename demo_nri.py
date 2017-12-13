from __future__ import print_function

import numpy as np
import nri_metrics as nm

# Define count table.
# First row and column are deletions and insertions, respectively
C = np.array([ [0,   100,    15,    10,   200],
               [10,    1,    10,   300,    20],
               [5,    10,   100,     5,    10] ])

print("\n=========================")
print("Results from nri()")
print("=========================")
[nriNet, precNet, recallNet, nriNeur, precNeur, recallNeur] = nm.nri(C)
print("nriNet = ",nriNet)
print("precNet = ",precNet)
print("recallNet = ",recallNet)

print("\n=========================")
print("Results from nri_slow()")
print("=========================")
[nriNet, precNet, recallNet, nriNeur, precNeur, recallNeur] = nm.nri_slow(C)
print("nriNet = ",nriNet)
print("precNet = ",precNet)
print("recallNet = ",recallNet)

print("")
