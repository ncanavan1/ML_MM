Submit the job smiles.job or execute gen_smiles.sh to generate the SMILES strings for the ligands described in appendix B, for the arrangements in page 16 of the dissertation document.

SMILES will be duplicated for each lanthinide in ln_list file

Converting from SMILES format to full molecules can is done by submitting the gen_geometry job. In this job script gen_geometry.sh is called, the parameter (10) is how many molecules randomly selected to convert to a full molecule. Larger numbers will mean larger computation times for everything in future. Molecules can be found in their .smi .xyz format as well as their cannonical format and .pdb format (better for visualisation). Note for this step chemaxon molconvert must be installed and visable to path. 
