  Codes
=================
- make_TTtable.f90: fortran-90 codes to make TT tables
	-No object file dependencies. To compile, simply type: gfortran -O make_TTtable.f90 -o make_TTtable
- make_TTtable.py: python driver to the fortran code, reads vz model from file
- make_TTtable.vz-fix.py: similar to above, but with hardwired velocity model (not read from file)



  Velocity models
==================
 - vzmodel.txt: used in ssprings example, for testing codes (assumed Vp/Vs = 1.732)
 - hk77model.txt: velocity model from Zach's ridgrecrest/ directory (assumed Vp/Vs = 1.68)
 - hkun07.sealevel.txt: from Hauksson and Unruh, 2007, which builds on Hauksson 2000 (assumed Vp/Vs = 1.70???), Z=0 at sea level
 - hkun07.elev1km.txt: same as above, but setting 1km above sea level as Z = 0
 - hkun07.elev2km.txt: same as above, but setting 2km above sea level as Z = 0

