# supercap_dataset

This is the dataset obtained by filterting QMOF for 2D structures. Some of these structures are denser than one might expect for MOFs - maybe an additional filter for largest poe diameter would be 
appropriate.


The Filtering directory contains the process I used to filter the QMOF database.

The Hartree_potentials directory contains the results of a HSE06 calculation of the pristine unit cell. Prior to this, I performed an oxidation state assignment with a machine learned model and did 
a spin ground state analysis based on the metal permutations. I had to discard some of the obtained MOFs due to convergence problems.
