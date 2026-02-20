import numpy

# Very basic system for guessing at the required resolution, 
# process grid, and runtime.
# This is for generating the job script, not the main_input file.
# This is far from perfect.  We base it purely on RaF.

def grid_pars(RaF):
    h36 ="36:00:00"  # wall time in hours:min:sec format
    h120 = "120:00:00"
    wt = h120
    #Note: in my initial estimates, I used h36 for the 3e4 case as well.
    if (RaF < 3e4):
        nr = 128
        ntheta = 192
        nprow = 32
        npcol = 32
    elif (RaF < 3e5):
        nr = 128
        ntheta = 384
        nprow = 64
        npcol = 32
    elif (RaF < 3e6):
        nr = 128
        ntheta = 768
        nprow = 64
        npcol = 32
    elif (RaF < 3e7):
        nr = 256
        ntheta = 1536
        nprow = 64
        npcol = 64
    else:
        return None # 
    
    return (nr,ntheta,nprow,npcol,wt)
    
#Wrapper routine that will create a directory if it doesn't exist
def emkdir(path):
    import os
    # Check if the main directory exists
    isExist = os.path.exists(path)

    if not isExist:  
      # Create a new directory because it does not exist 
      os.makedirs(path)

#Job-script generator.  This works only for Pleiades currently.
def gen_job_script(ofile,ncpu,nprow,npcol,walltime, mnum, template='base_script.txt',
                   rayleigh_exec = 'rayleigh.avx', node_cpu=20,node_type='ivy', prefix = 'model'):
    #first, read the base script file that we will prepend and append to create
    #the jobscript file
    base_fo = open(template, "r")
    base_lines = []
    while True:                            # Keep reading forever
        theline = base_fo.readline()   # Try to read next line
        if len(theline) == 0:              # If there are no more lines
            break                          #     leave the loop
        else:
            base_lines.append(theline)
        # Now process the line we've just read

    base_fo.close()

        
    ##############################################################
    # Add the PBS commands
    pbs = []
    mname = '#PBS -N '+prefix+'_'+str(mnum)+'\n'
    pbs.append(mname)
    wstring = '#PBS -l walltime='+walltime+'\n'
    pbs.append(wstring)
    
    nselect = ncpu//node_cpu
    test = nselect*node_cpu
    if (test < ncpu):
        nselect +=1
    

    select_str ="#PBS -l select="+str(nselect)
    select_str+=":ncpus="+str(node_cpu)
    select_str+=":mpiprocs="+str(node_cpu)
    select_str+=":model="+node_type+"\n"
    
    pbs.append(select_str)
    
    ############################################################
    #  Now the main body of the script

    body = ["\n"]
    body.append("module load comp-intel\n")
    body.append("module load mpi-hpe\n")
    body.append("export OMP_NUM_THREADS=1\n\n")

    body.append("ln -s /home4/nfeather/Ra_share/Rayleigh/bin/* .\n")
    body.append("sleep 10\n\n")

    mpicmd = "mpiexec -np "+str(ncpu)+" ./"+rayleigh_exec+" -nprow "+str(nprow)+" -npcol "+str(npcol)+"\n"
    body.append(mpicmd)

    
    #############################################
    # Build the new input file data
    flines = []
    for b in base_lines:
        flines.append(b)
    for p in pbs:
        flines.append(p)
    for b in body:
        flines.append(b)
    #for b in base_lines:
    #    flines.append(b)
    #for rl in rlines:
    #    flines.append(rl)
    #for tl in tn:
    #    flines.append(tl)
        

    write_file = True
    if (write_file):
        # Write the file    
        ofo = open(ofile, "w")
        for  fl in flines:
            ofo.write(fl)
        ofo.close()
    else:
        for fl in flines:
            print(fl,end='')
def nondimensional_parameters(ecoefs, verbose = False):
    #
    # Calculate nondimensional numbers associated with 
    # a given equation_coefficients object.
    #
    # When calculating RaF, this version currently assumes internal heating with 
    # no imposed entropy/temperature gradient at the boundaries.
    
    # Rayleigh's arrays are stored in descending order in radius.
    # We do a lot of reversing arrays so that
    # so that NumPy's trapz routine is happy. 
    
    radius = ecoefs.radius[::-1]
    rho = ecoefs.functions[0,:][::-1]  # background density
    T = ecoefs.functions[3,:][::-1]    # background temperature
    Q = ecoefs.functions[5,:][::-1]    # volumetric heating term
    
    #########################################################
    # Note, this next bit is a little tricky. We want g, the
    # local graviational acceleration, but the equation
    # coefficients file really stores g*rho/cp. We can divide
    # out rho, but we can't disentangle g from cp, but that's OK.
    # They always come together as g/cp
    g_over_cp = ecoefs.functions[1][::-1]/rho
    ########################################################

    nu = ecoefs.constants[4]*ecoefs.functions[2,:][::-1]
    kappa = ecoefs.constants[5]*ecoefs.functions[4,:][::-1]
    omega = ecoefs.constants[0]/2

    
    L = numpy.max(radius)-numpy.min(radius)

    # Build the Flux associated with the internal heating
    # (equation 12 of Featherstone and Hindman, 2016, ApJ, 818, 32
    lumr = run_int(Q,radius)*ecoefs.constants[9]
    Flux = lumr/(4*numpy.pi*radius**2)  # This is the flux the convection must carry at reach radius

    if (verbose):
        # Run a check on the integration routines
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('As a check of the integration routines, vol_int(Q,radius) should be 1 to within precision.')
        print('vol_int(Q,radius): ', vol_int(Q,radius))
        print('')

    # Now create our volume averages
    rho_avg   = vol_avg(rho,radius)
    nu_avg    = vol_avg(nu,radius)
    kappa_avg = vol_avg(kappa,radius)
    T_avg     = vol_avg(T,radius)
    F_avg     = vol_avg(Flux,radius)
    g_over_cp_avg = vol_avg(g_over_cp,radius)

    # Compute RaF (eq. 14 of Featherstone and Hindman 2016)
    num        = g_over_cp_avg*F_avg*L**4
    denom      = rho_avg*T_avg*(nu_avg*kappa_avg*kappa_avg)
    ecoefs.Raf = num/denom

    # Compute Ekman
    ecoefs.Ek = nu_avg/L/L/omega

    # Compute Convective Rossby
    ecoefs.Roc = numpy.sqrt(ecoefs.Raf*ecoefs.Ek*ecoefs.Ek)    
    
    # Compute Prandtl
    ecoefs.Pr = nu_avg/kappa_avg
    
    ecoefs.nu = nu
    ecoefs.nu_avg = nu_avg

    
def get_new_dim_pars(eq,Raf_new,Pr_new, Ek_new):
    """Based on equation state contained in the equation_coefficients object, eq,
       returns (nu_top, kappa_top, Omega)
       required to yield specified Raf, Pr and Ek.
       The values returned are only appropriate if the functional form of nu(r)
       and kappa(r) to be used is the same as that contained in eq."""
    nondimensional_parameters(eq) # The equation coefficients

    L = numpy.max(eq.radius)-numpy.min(eq.radius) # Shell Depth
    Raf_ref   = eq.Raf  # parameters specific to this particular model
    Pr_ref    = eq.Pr
    nu_ref    = eq.constants[4]
    kappa_ref = eq.constants[5]
    
    Ra_ratio = (Raf_new*Pr_new)/(Raf_ref*Pr_ref)
    nu_top = nu_ref*(Ra_ratio)**(-1/3)
    Omega = nu_top/L/L/Ek_new
    
    kappa_top = nu_top/Pr_new
    
    return(nu_top,kappa_top,Omega)
    
def vol_avg(f,radius):
    # computes volume average of (f(r))
    integrand = f*radius**2
    num = numpy.trapz(integrand,radius)
    denom = numpy.trapz(radius**2,radius)
    return num/denom
def vol_int(f,radius):
    # returns 4*pi * int_{rmin}^{rmax} (f(r)*r^2)
    integrand = f*radius**2
    val = numpy.trapz(integrand,radius)*4*numpy.pi
    return val
def run_int(f,radius):
    # returns g(r) where 
    # g(r) = 4*pi*int_{rmin}^r [f(r)*r^2]
    nr = len(radius)
    g = numpy.zeros(nr,dtype='d')
    g[0] = 0
    for i in range(1,nr):
        fint = f[0:i+1]
        rint = radius[0:i+1]
        g[i] = vol_int(fint,rint)
    return g