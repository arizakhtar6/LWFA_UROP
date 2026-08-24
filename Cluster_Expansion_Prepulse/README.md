This folder contains all files relating to my investigation of how Methane (CH4) clusters expand under the laser pre-pulse measured in the Gemini North experiment in the Central Laser Facility (CLF).

All input files in this project are run using Smilei, in 2D. To run any namelists, you will need to have successfully compiled Happi (refer to https://smileipic.github.io/Smilei/Use/installation.html) and you will need the Happi folder in the directory you are running the namelist from. 

For the pre-pulse time envelope, you will need to read in the data from the Gemini North pre-pulse measurement (attached in a csv file titled 'gemini_prepulse_envelope.csv'), within which I have specifically extracted the normalised intensity and time (time before the main pulse is negative).

I have also added an example submission script that can be used to run the namelists on the HPC. Specifying OmpThreads (especially in the 3rd line) gets it on the cluster much faster, as it reads this line in its initial pass of the pbs file. So if you don't specify it here, you'll notice it takes a lot longer to run because it logs the job as one that requires resources without any multi-threading.
