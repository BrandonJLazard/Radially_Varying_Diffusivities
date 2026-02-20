import numpy as np
import numpy
maxq =4000
class Shell_Spectra:
    """Rayleigh Shell Spectrum Structure
    ----------------------------------
    self.niter                                    : number of time steps
    self.nq                                       : number of diagnostic quantities output
    self.nr                                       : number of shell slices output
    self.nell                                     : number of ell values
    self.nm                                       : number of m values
    self.lmax                                     : maximum spherical harmonic degree l
    self.mmax                                     : maximum spherical harmonic degree m
    self.qv[0:nq-1]                               : quantity codes for the diagnostics output
    self.radius[0:nr-1]                           : radii of the shell slices output
    self.inds[0:nr-1]                             : radial indices of the shell slices output
    self.vals[0:lmax,0:mmax,0:nr-1,0:nq-1,0:niter-1] 
                                                  : The complex spectra of the shells output 
    self.lpower[0:lmax,0:nr-1,0:nq-1,0:niter-1,3]    : The power as a function of ell, integrated over m
                                                     :  index indicates (0:total,1:m=0, 2:total-m=0 power)
    self.iters[0:niter-1]                         : The time step numbers stored in this output file
    self.time[0:niter-1]                          : The simulation time corresponding to each time step
    self.version                                  : The version code for this particular output (internal use)
    self.lut                                      : Lookup table for the different diagnostics output
    self.time_averaged                            : [True or False] Indicates if power has been time-averaged

    Initialization Examples:
        (1):  Read in a single Shell_Spectra file
              a = Shell_Spectra('00000001',path='./Shell_Spectra/')
              
        (2):  Time-average power (i.e. average over all records) in single Shell_Spectra file:
              a = Shell_Spectra(['00000001'])   {filename is contained in brackets}
              
        (3):  Time-average power in several Shell_Spectra files + save data to new Shell_Spectra file:
              a = Shell_Spectra(['00000001','00000002'],ofile='my_save_file.dat')
              
        (4):  Time-average over last 10 files in long list of files:
              a = Shell_Spectra(file_list,nfiles=10)
              
        (5):  Time-average of last 0.5 units of simulation time from long list of files
              (units depend on simulation setup)
              a = Shell_Spectra(file_list,dt=0.5)
              
        (6):  Read in a Shell_Spectra file, extract only quantity codes 2 and 3:
              a = Shell_Spectra(infile,qcodes=[2,3])  # Works with time-averaging mode as well
              
    Additional Notes:
        For Time-averaged data:
        
            - Output files are written in a similar format to a regular Shell_Spectra file. They may be
              read in as if they were a normal Shell_Spectra file.
             
            - The data structure contains only data for a single 'time step' (the time average), but
              iters and time contain two values, the iteration numbers and times corresponding to the 
              beginning and end of the time average.
             
            - The version number is listed as 100+the Rayleigh version number.
            
            - Time-averaging is performed the power (a quadratic function).  The sqrt of the power,
              (i.e., the amplitude of the complex spectral coefficient) is stored in the real portion
              of vals.  The complex portion of vals is set to zero.
              
            - The time_averaged attribute is set to True for time-averaged data.

    """

    def print_info(self):
        """ Prints all metadata associated with the shell-spectra object."""
        print( 'version  : ', self.version)
        print( 'niter    : ', self.niter)
        print( 'nq       : ', self.nq)
        print( 'nr       : ', self.nr)
        print( 'nell     : ', self.nell)
        print( 'nm       : ', self.nm)
        print( 'lmax     : ', self.lmax)
        print( 'mmax     : ', self.mmax)
        print( '.......................')
        print( 'radius   : ', self.radius)
        print( '.......................')
        print( 'inds     : ', self.inds)
        print( '.......................')
        print( 'iters    : ', self.iters)
        print( '.......................')
        print( 'time     : ', self.time)
        print( '.......................')
        print( 'qv       : ', self.qv)


    def __init__(self,filename='none',path='Shell_Spectra/', ofile='none', qcodes=[], dt=-1,nfiles=-1):
        """
           filename  : (a) {string} The Shell_Spectra file to read.
                     : (b) {list of strings} The Shell_Spectra file(s) whose power
                     :     you wish to time average.
           path      : The directory where the file is located (if full path not in filename)
           qcodes    : {optional; list of ints} Quantity codes you wish to extract (if not all)
           ofile     : {optional; string} Filename to save time-averaged data to, if desired.
           nfiles    : optional -- number of files to read relative to last file 
                                   in the list (time-averaging mode only)
           dt        : optional -- maximum time to average over, relative to
                                   last iteration of last file in the list
                                   (time-averaging mode only)
        """

        ############################################################    

        # Check to see if we are time-averaging
        # This is true if (a) ofile is set or (b) filename is a list of files, rather than a single string
        # if (a) is false, but (b) is true, no output is written, but time-averaged
        # output is returned
        
        #(a)
        time_average_the_data = False
        mainfile=filename
        if (ofile != 'none'):
            time_average_the_data = True
            
        #(b)
        ltype = type([])        
        ftype=type(filename)
        if (ltype == ftype):
            time_average_the_data = True
        else:
            if (mainfile == 'none'):
                mainfile = path+'00000001'
            else:
                mainfile = path+mainfile           
            
        if(time_average_the_data):    
            mainfile=filename[0]

        # Read the header of mainfile, whether time-averaging or not
        fd = open(mainfile,'rb')
        # We read an integer to assess Endianness
        bs = check_endian(fd,314,'int32')
        self.read_header(fd,bs)

        ########################################################
        # We may want to extract a subset of the quantity codes
        if (len(qcodes) == 0):
            qget = self.qv
        else:
            qget = numpy.array(qcodes,dtype='int32')
            self.qv = qget  # Don't update the lookup table yet

        nqfile = self.nq        # number of quantity codes in the file
        self.nq = len(self.qv)  # number of quantity codes we will extract

        # Version numbers greater than 100 indicate a 
        # that the data has been time-averaged, so we read
        # and extra time and iteration number at the end.
        
        self.time_averaged = (self.version > 100)
            
        if (time_average_the_data):
            self.time_averaged=True
            fd.close()
            self.time_average(filename,dt=dt,nfiles=nfiles)
            
            if (ofile != None):
                self.write(ofile)
        else:            
            #############################################################

            self.vals  = np.zeros((self.nell,self.nm,self.nr,self.nq,self.niter),dtype='complex128')

            if (self.time_averaged):
                self.iters = np.zeros(2,dtype='int32')
                self.time  = np.zeros(2,dtype='float64')    
            else:
                self.iters = np.zeros(self.niter,dtype='int32')
                self.time  = np.zeros(self.niter,dtype='float64')

            dcount=self.nr*self.nell*self.nm*nqfile
            shape_tuple=(self.nm,self.nell,self.nr,nqfile)

            for i in range(self.niter):
                #For now, read all data.  
                #Could get a little more efficient by using seek.

                # Real part first (all q's are grouped together)
                tmp = np.reshape(swapread(fd,dtype='float64',count=dcount,swap=bs),shape_tuple, order = 'F')

                for q in range(self.nq):
                    #print('checking: ',self.qv)
                    qcheck = self.lut[qget[q]]
                    if (qcheck < maxq): 
                        self.vals[:,:,:,q,i].real = tmp[:,:,:,self.lut[qget[q]]]

                # Imaginary part second (again, all q's grouped)
                if (not self.time_averaged):
                    tmp2 = np.reshape(swapread(fd,dtype='float64',count=dcount,swap=bs),shape_tuple, order = 'F')
                    for q in range(self.nq):
                        qcheck = self.lut[qget[q]]
                        if (qcheck < maxq): 
                            self.vals[:,:,:,q,i].imag = tmp2[:,:,:,self.lut[qget[q]]]

                self.time[i] = swapread(fd,dtype='float64',count=1,swap=bs)
                self.iters[i] = swapread(fd,dtype='int32',count=1,swap=bs)

            if (self.version != 4 and (not self.time_averaged)):
                # The m>0 --power-- is too high by a factor of 2
                # We divide the --complex amplitude-- by sqrt(2)
                self.vals[:,1:,:,:,:] /= np.sqrt(2.0)

            

            if (self.time_averaged):
                self.time[1] = swapread(fd,dtype='float64',count=1,swap=bs)
                self.iters[1] = swapread(fd,dtype='int32',count=1,swap=bs)            
            fd.close()

            
        self.build_power()
        self.lut = get_lut(self.qv) #Rebuild this in case qcodes were specified         
                        
    def time_average(self, file_list, dt=-1,nfiles=-1):
        import numpy as np
        """
            file_list -- list of Shell_Spectra files (strings) to average over.
            
            nfiles -- optional -- number of files to read relative to last file 
                                  in the list
            dt     -- optional -- maximum time to average over, relative to
                                  last iteration of last file in the list       
        """

        num_files = len(file_list)
        flast = -1
        if (nfiles > 0):
            flast = num_files-nfiles-1
        if (flast < 0):
            flast = -1

        nfiles = len(file_list)
        a = Shell_Spectra(file_list[nfiles-1], path = '')
        nr = a.nr
        nell = a.nell
        nm = a.nm

        self.vals = np.zeros((nell, nm, nr,self.nq,1),dtype='complex128')

        i0 = np.zeros(1,dtype='int32')
        t0 = np.zeros(1,dtype='float64')

        ifinal = np.zeros(1,dtype='int32')
        tfinal = np.zeros(1,dtype='float64')    

        qcount = np.zeros(self.nq,dtype='int32')

        ifinal[0] = a.iters[a.niter-1]
        tfinal[0] = a.time[a.niter-1]

        for i in range(num_files-1,flast,-1):
            the_file = file_list[i]
            print(the_file)
            b = Shell_Spectra(the_file,path='',qcodes=self.qv)
            nrec = b.niter
            br = b.nr
            bell = b.nell
            if ((br == nr) and (bell == nell)):
                for j in range(nrec):
                    # Time and iter check
                    time_check = True
                    if (dt > 0):
                        dt0 = tfinal-b.time[j]
                        if (dt0 > dt):
                            time_check = False

                    if (time_check):
                        for q in range(self.nq):
                            qcheck = b.lut[self.qv[q]]
                            if (qcheck < maxq):
                                # Note:  We actually time-average the power, not the spectra.
                                self.vals[:,:,:,q,0] += np.real(b.vals[:,:,:,qcheck,j])**2
                                self.vals[:,:,:,q,0] += np.imag(b.vals[:,:,:,qcheck,j])**2
                                qcount[q]+=1

                        t0[0] = b.time[j]
                        i0[0] = b.iters[j]
        
        # Average the POWER of each q-code independently 
        # (in case outputs changed mid run)
        for q in range(self.nq):
            div = qcount[q]
            if (div > 0):
                self.vals[:,:,:,q] /=div
 
        ############################################
        # A few parts of the data structure need to 
        # be modified when time averaging
        self.vals = numpy.sqrt(self.vals)  
        self.iters = numpy.zeros(2,dtype='int32')
        self.iters[0] = i0
        self.iters[1] = ifinal
        self.time  = numpy.zeros(2,dtype='float64')
        self.time[0] = t0
        self.time[1] = tfinal
        self.niter = 1
        self.version=a.version+100  # versions > 100 --> time-averaged
        ###################################


    def write(self,ofile):


        ndim=6

        fd = open(ofile,'wb') #w = write, b = binary
        dims = np.zeros(ndim,dtype='int32')
        dims[0] = 314
        dims[1] = self.version
        dims[2] = 1
        dims[3] = self.lmax
        dims[4] = self.nr
        dims[5] = self.nq

        dims.tofile(fd)
        self.qv.tofile(fd)

        self.radius.tofile(fd)
        self.inds.tofile(fd)
        for i in range(self.niter):
            tmp = np.transpose(np.real(self.vals[:,:,:,:,i]))
            tmp.tofile(fd)
            if (not self.time_averaged):
                tmp = np.transpose(np.real(self.vals[:,:,:,:,i]))
                tmp.tofile(fd)                
            self.time[i].tofile(fd)
            self.iters[i].tofile(fd)
        # The final structure is identical to a normal az_average file save for the fact that final iteration adn final time are saved
        if (self.time_averaged):
            self.time[1].tofile(fd)
            self.iters[1].tofile(fd)
        fd.close()
            
    def build_power(self):
            ####            

        self.lpower  = np.zeros((self.nell,self.nr,self.nq,self.niter,3),dtype='float64')
        #!Finally, we create the power
        for k in range(self.niter):
            for q in range(self.nq):
                for j in range(self.nr):
                    # Load the m=0 power
                    self.lpower[:,j,q,k,1] += numpy.real(self.vals[:,0,j,q,k])**2
                    self.lpower[:,j,q,k,1] += numpy.imag(self.vals[:,0,j,q,k])**2

                    # m !=0 (convective) power
                    for m in range(1,self.nm):
                        self.lpower[:,j,q,k,2] += numpy.real(self.vals[:,m,j,q,k])**2
                        self.lpower[:,j,q,k,2] += numpy.imag(self.vals[:,m,j,q,k])**2

        self.lpower[:,:,:,:,0] = self.lpower[:,:,:,:,2]+self.lpower[:,:,:,:,1] # total power

    def read_header(self,fd,bs):
        version = swapread(fd,dtype='int32',count=1,swap=bs)
        nrec = swapread(fd,dtype='int32',count=1,swap=bs)
        lmax = swapread(fd,dtype='int32',count=1,swap=bs)
        nell = lmax+1
        nm = nell   
        mmax = nm-1
        nr = swapread(fd,dtype='int32',count=1,swap=bs)
        nq = swapread(fd,dtype='int32',count=1,swap=bs)

        self.niter = nrec
        self.nq = nq
        self.nr = nr
        self.nell = nell
        self.nm   = nm
        self.lmax = lmax
        self.mmax = mmax

        self.qv = np.reshape(swapread(fd,dtype='int32',count=nq,swap=bs),(nq), order = 'F')
        self.radius = np.reshape(swapread(fd,dtype='float64',count=nr,swap=bs),(nr), order = 'F')
        self.inds = np.reshape(swapread(fd,dtype='int32',count=nr,swap=bs),(nr), order = 'F')        
        self.version = version

        # convert from Fortran 1-based to Python 0-based indexing
        self.inds = self.inds - 1
        # Build the lookup table
        self.lut = get_lut(self.qv)
        
def check_endian(fd,sig,sigtype):
    # returns False if first element read from file matches sig
    # True otherwise
    chk = np.fromfile(fd,dtype=sigtype,count=1)
    if (chk == sig):
        return False
    else:
        return True

maxq = 4000

def get_lut(quantities):
    """return the lookup table based on the quantity codes"""
    nq = len(quantities)
    lut = np.zeros(maxq) + maxq
    for i,q in enumerate(quantities):
        if ((0 <= q) and ( q <= maxq-1)): # quantity must be in [0, maxq-1]
            lut[q] = i
    return lut.astype('int32')
    
def swapread(fd,dtype='float64',count=1,swap=False):
        #simple wrapper to numpy.fromfile that allows byteswapping based on Boolean swap
        if (swap):
                val = np.fromfile(fd,dtype=dtype,count=count).byteswap()
        else:
                val = np.fromfile(fd,dtype=dtype,count=count)
        if (len(val) == 1):
                val = val[0]
        return val
