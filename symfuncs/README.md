To generate the over full list of symmetry functions, execute the gen_sf.py file, output will be file containing all symfunctions.
Parameters of symmetry functions to be generated  may be altered in the gen_sf.py file.

To determine the value of all symmetry functions on each molecule in the molecules directory, submit the sf_eval.job. To compile sfeval.cpp, use the "g++ sfeval.cpp -fopenmp -o sfeval". The output will be saved to a csv

Once this job is complete, execute "./FPS.py <full_sf_csv_file> <energy_csv> <features_to_reduce_to> <name_for_reduced_file>", which will complete feature space reduction via FPS. Output will be in the form of a CSV containing sf values and asscoicated energies for each molecule. Aditionally, a file containing the index of the reduced set of symmetry functions will be given in a text file, where the index is the position in the original list of symmetry functions. This has potential to reduce time for sf_eval.job in future. 
