#! /bin/env python

##### Python driver script to generate a travel-time table TT(X,Z) based on a predefined
##### velocity model. The travel time table can be for direct P or S (Pg, Pg) and the velocity
##### model can be for flat earth V(z) or radial earth V(r).
##### 
##### The ray tracing computation uses the precompiled Fortran90 source code make_TTtable.f90

##### -----------------   Daniel Trugman, 2019 --------------------------- ###############

### import statements
import subprocess

### script setup, program arguments

#--- path to the Fortran90 executable
src_file = "./make_TTtable"

#--- define travel time table arguments
phases = [1,2]                            # list of phase codes (1 = P, 2 = S)
tt_outformat = 1                          # output format: 1 = (x in km, t in sec), 2 = (x in deg, t in min.)
tt_dep0, tt_dep1, tt_ddep = 0., 200., 3.   # travel-time table depths (km):  min, max, space
tt_del0, tt_del1, tt_ddel = 0., 400., 4.  # travel-time table offsets (km):  min, max, space
vpvs_factor = 1.73#1.732                       # Vp/Vs ratio (only used if Vs is unlisted in velocity model
ray_params_min = [.133, .238]             # min ray param at long range to prevent refracted phases
                                          # (.133 = no Pn, .238 = no Sn)

#--- define velocity model arguments
vmdl_name = "LA_Bie_SRL_2020.txt"      # name of velocity model to read  
vmdl_informat  = 1  # velocity model input format (1: depth, 2:radius)
#-----
#      The input model has one line per depth point: (z,Vp,Vs).
#      The program interpolates between the depth points, so models with constant velocity
#      layers must include both the layer top and bottom as depth points for each layer.
#      If Vs = 0., the program infers it using Vp and vp_vsfactor

# --- read model from file
vmdl = ""
npts = 0 # number of depth points
with open(vmdl_name,'r') as f:
	for line in f:
		vmdl+=line
		npts += 1

# --- loop over phases
for iphase in phases:

	# --- define output file
	ttfile = 'TT.' + vmdl_name[:-3]
	if iphase ==1:
		ttfile += 'pg'
	elif iphase == 2:
		ttfile += 'sg'
	else:
                print('Error, undefined phase type')
                raise KeyboardInterrupt

        # select minimum ray parameter, depending on phase
	ray_param_min = ray_params_min[iphase-1]

	#---- convert arguments into the proper format for the Fortran90 executable
	args = ""
	args+= "{:}\n".format(ttfile)
	args+= "{:}\n".format(iphase)
	args+= "{:}\n".format(tt_outformat)
	args+= "{:} {:} {:}\n".format(tt_dep0, tt_dep1, tt_ddep)
	args+= "{:} {:} {:}\n".format(tt_del0, tt_del1, tt_ddel)
	args+= "{:}\n".format(vpvs_factor)
	args+= "{:}\n".format(ray_param_min)
	args+= "{:}\n".format(vmdl_name)
	args+= "{:}\n".format(vmdl_informat)
	args+= "{:}\n".format(npts)
	args+= vmdl

	#----- call Fortran90 executable to make travel time table
	#      (run notes should now appear in the terminal)
	out = subprocess.run([src_file],input=args,text=True)
	print(out)

# finished all phases
print ("Done with all phases.")
	
