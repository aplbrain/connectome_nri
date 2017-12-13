import numpy as np



def nri(C):
   # This is a relatively fast implementation, but the code may be difficult
   # to understand since matrix operations have been leveraged for acceleration.
   # See the nri_slow() function for a version that is easier to read
   # and understand.

   # Computes the Neural Reconstruction Index (NRI). Given a count table of
   # matching synaptic terminals, C (matched by location and polarity), for a
   # reference (ground truth) and test (automated segmentation)
   # reconstruction, calculate NRI for individual neurons (local scores) and
   # the full network (global score).

   # The ith row and jth column of C should contain the number of
   # matching terminals for the ith reference neuron/object and the jth test
   # neuron/object. The 0th row and 0th column are not-founds (deletions in
   # (i,0) and insertions in (0,j)). The (0,0) entry of C should always be
   # 0 since a synapse cannot be both deleted and inserted.

   # OUTPUTS
   # -------
   # nriNet:      A scalar value that is the global/network NRI score
   # precNet:     A scalar value that is the global/network Precision score
   # recallNet:   A scalar value that is the global/network Recall score
   # nriNeur:     A vector of NRI scores, one for each neuron. Values are ordered
   #              to match rows of the count table, C. The first value nriNeur[0]
   #              is set to NaN, since that row of C represents insertions, not a
   #              reference neuron.
   # precNeur:    A vector of Precision scores, one for each neuron. 
   # recallNeur:  A vector of Recall scores, one for each neuron. 

   # Matt Roos
   # JHU/APL
   # 2/14/17

   C = C.astype(float)
   
   # TP count for a neuron is sum of cij*(cij-1)/2, summed
   # over all j (all elements in a row excluding the first column, ci0)
   Z = C*(C-1)/2
   tpNeur = np.sum(Z[:,1:],axis=1)
   tpNeur[0] = 0 # don't count TPs in insertion row

   # FP count includes the sum of all possible products, cij*cpj, where the
   # sum excludes terms with p=i, or j=1.  The sum is divided by two so the
   # FPs aren't counted twice (i.e., once for each of two merged neurons). The
   # FP count also includes c1j-choose-2 summed over all j>1 (FPs due to pairs
   # of inserted terminals in row 1).
   FPij = (np.sum(C,axis=0,keepdims=True) - C) * C/2
   FPij[0,:] = FPij[0,:] + Z[0,:]
   fpNeur = np.sum(FPij[:,1:],axis=1)


   # FN count includes [1] ci1*(ci1-1)/2 where ci1 is number of deleted
   # synapses and [2] all possible products, cij*cik (where cij and cik are
   # jth and kth elements of the ith row), excluding k,j=1 (the deletion
   # column) and j>=k (that is, we include cij*cik but not cik*cij and not
   # cij*cij).
   # Division by two in line below is because cij*cik and cik*cij are both
   # counted, but only one is wanted (i.e., j>=k)
   FNij = (np.sum(C,axis=1,keepdims=True) - C) * C/2
   fnNeur = np.sum(FNij,axis=1) + Z[:,0]
   fnNeur[0] = 0; # don't count FNs in insertion row


   # Compute NRI, precision, and recall for individual neurons
   with np.errstate(invalid='ignore'):
      precNeur = tpNeur / [tpNeur+fpNeur]
      recallNeur = tpNeur / [tpNeur+fnNeur]
      nriNeur = 2*tpNeur / (2*tpNeur + fpNeur + fnNeur) # same at 2*P*R/(P+R) but without undefined P or R problem
   nriNeur[0] = np.nan # insertion row, not a ground truth neuron


   # Compute NRI for full network
   TP = np.nansum(tpNeur)
   FP = np.nansum(fpNeur)
   FN = np.nansum(fnNeur)
   precNet = TP / (TP + FP)
   recallNet = TP / (TP + FN)
   nriNet = 2*TP / (2*TP + FP + FN) # same at 2*P*R/(P+R) but without undefined P or R problem

   return (nriNet, precNet, recallNet, nriNeur, precNeur, recallNeur)



def nri_slow(C):
   # This is a very slow implementation, but is intended to be easy to read
   # and understand.

   # Computes the Neural Reconstruction Index (NRI). Given a count table of
   # matching synaptic terminals, C (matched by location and polarity), for a
   # reference (ground truth) and test (automated segmentation)
   # reconstruction, calculate NRI for individual neurons (local scores) and
   # the full network (global score).

   # The ith row and jth column of C should contain the number of
   # matching terminals for the ith reference neuron/object and the jth test
   # neuron/object. The 0th row and 0th column are not-founds (deletions in
   # (i,0) and insertions in (0,j)). The (0,0) entry of C should always be
   # 0 since a synapse cannot be both deleted and inserted.

   # OUTPUTS
   # -------
   # nriNet:      A scalar value that is the global/network NRI score
   # precNet:     A scalar value that is the global/network Precision score
   # recallNet:   A scalar value that is the global/network Recall score
   # nriNeur:     A vector of NRI scores, one for each neuron. Values are ordered
   #              to match rows of the count table, C. The first value nriNeur[0]
   #              is set to NaN, since that row of C represents insertions, not a
   #              reference neuron.
   # precNeur:    A vector of Precision scores, one for each neuron. 
   # recallNeur:  A vector of Recall scores, one for each neuron. 

   # Matt Roos
   # JHU/APL
   # 2/14/17
   
   [I, J] = C.shape;  # I-1 reference neurons, J-1 test neurons

   tpNeur = np.zeros([I,1])
   fnNeur = np.zeros([I,1])
   fpNeur = np.zeros([I,1])

   # Loop over all the reference neurons (rows, excluding 0th row)...
   for iRef in range(1,I):
      # In comments below, cij is the (i,j)th element of C

      # TP count for a neuron is sum of cij*(cij-1)/2 where i=iRef, summed
      # over all j (all elements in a row excluding the first column, ci0)    
      tpNeur[iRef] = np.sum(C[iRef,1:]*(C[iRef,1:]-1)/2)


      # FN count includes [1] ci0*(ci0-1)/2 where ci0 is number of deleted
      # synapses and [2] all possible products, cij*cik (where cij and cik are
      # jth and kth elements of the iRef row), excluding k,j=0 (the deletion
      # column) and j>=k (that is, we include cij*cik but not cik*cij and not
      # cij*cij).
      fnNeur[iRef] = C[iRef,0]*(C[iRef,0]-1)/2

      for jTest in range(0,J-1):
         for kTest in range(jTest+1,J):
            fnNeur[iRef] += C[iRef,jTest]*C[iRef,kTest]

      # FP count includes the sum of all possible products, cij*cpj, where
      # i=iRef and the sum excludes term with p=i, or j=0.  The count also
      # needs a value for i=iRef=0 (insertion row), which is outside of
      # (after) this iRef loop.
      for pRef in range(0,I):
         for jTest in range(1,J):
            if pRef != iRef:
               # All pairs of inserted and true terminals are FPs (split between
               # the ground truth neuron and the "insertion neuron")
               fpNeur[iRef] += C[iRef,jTest]*C[pRef,jTest]/2;

   # A FP count is also needed for the insertion (0th) row, in order for the
   # sum of the neuron FPs to equal that of the true network FP
   iRef = 0;   # the insertion row, not really a reference neuron
   for pRef in range(0,I):
      for jTest in range(1,J):
         if pRef == iRef:
            # All pairs of inserted terminals [C(i,j)-choose-2] are FPs
            fpNeur[iRef] += C[iRef,jTest]*(C[iRef,jTest]-1)/2;  
         else:
            # All pairs of inserted and true terminals are FPs (split between
            # the ground truth neuron and the "insertion neuron")
            fpNeur[iRef] += C[iRef,jTest]*C[pRef,jTest]/2;


   # Compute NRI, precision, and recall for individual neurons
   with np.errstate(invalid='ignore'):
      precNeur = tpNeur / [tpNeur+fpNeur]
      recallNeur = tpNeur / [tpNeur+fnNeur]
      nriNeur = 2*tpNeur / (2*tpNeur + fpNeur + fnNeur) # same at 2*P*R/(P+R) but without undefined P or R problem
   nriNeur[0] = np.nan # insertion row, not a ground truth neuron


   # Compute NRI for full network
   TP = np.nansum(tpNeur)
   FP = np.nansum(fpNeur)
   FN = np.nansum(fnNeur)
   precNet = TP / (TP + FP)
   recallNet = TP / (TP + FN)
   nriNet = 2*TP / (2*TP + FP + FN) # same at 2*P*R/(P+R) but without undefined P or R problem

   return (nriNet, precNet, recallNet, nriNeur, precNeur, recallNeur)
   
