SurfaceGrowthCrossPlatform

SurfaceGrowthCrossPlatform - a CUDA-based code for classical molecular dynamics simulations.

This is an updated version of SurfaceGrowthConsole that can be built on Windows and Linux (Ubuntu)
with some additional features (e.g. an ability to handle > 1 M atoms).

_______________________________________________________________________
Input file should have the name 'sugr_in.txt' and be in the same directory as the executable. 

Output is also saved in that directory: 
1) Diffuse.txt - time dependencies of the diffusion constant.
2) sg_...txt - measurements of different quantities.
3) rdf_...txt - radial distribution function.
4) sugr_...pdb - data for visualization in vmd.
