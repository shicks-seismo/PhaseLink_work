! ============================== PROGRAM MAKE_TTTABLE ================================== !

! This program is a self-contained Fortran90 script to generate a travel time table T(X,Z)
! for P or S arrivals from a source located at a depth Z to point at the surface (Z = 0)
! at horizontal offset X. The calculated travel times are obtained via numerical 
! ray tracing through input velocity model. 

! No object file dependencies. To compile, simply type: gfortran -O make_TTtable.f90 -o make_TTtable

program make_TTtable
	
	implicit none
	
	! ---------------- parameter and variable declarations  ---------------
	integer, parameter :: npts0=1000,nz0=100,nx0=500,nray0=40002,ncount0=80002
	real, parameter :: erad=6371., pi=3.1415927
	!-------
	integer :: i,j,k,kk,iz,i2,ideprad
    real :: zz,z1,z2,fact,zmin,zmax,alpha0,beta0,sfact,dz
	integer :: npts,ndep,nump,np,ncount,ndel,nz_s,nz,nk
	integer :: ideptype,idep,itype,icount,idel,iw,imth,irtr
	real :: ecircum,kmdeg,degrad,angle
	real :: p,pmin,pmax,pstep,plongcut,frac,xcore,tcore,h
	real :: dep,dep1,dep2,dep3,del,del1,del2,del3,deldel
	real :: scr1,scr2,scr3,scr4,xold,x,x1,x2,dx,t,t1,t2,dt
	real :: tbest,pbest,ubest
	real :: xsave(ncount0),tsave(ncount0),psave(ncount0),usave(ncount0)
	real :: deptab(nz0),ptab(nray0),delttab(nray0)
	real :: z_inp(npts0),alpha_inp(npts0),beta_inp(npts0) 
	real :: z_s(npts0),r_s(npts0),alpha_s(npts0),beta_s(npts0)
	real :: z(npts0),alpha(npts0),beta(npts0)
	real :: slow(npts0,2),deltab(nray0),tttab(nray0)
	real :: angang(nx0,nz0),tt(nx0,nz0),rayray(nx0,nz0),etaeta(nx0,nz0)
	real :: depxcor(nray0,nz0),deptcor(nray0,nz0)
	real :: depucor(nray0,nz0)
    character*100 ttfile, vmodel

    ! define fixed variables
	ecircum=2.*pi*erad
	kmdeg=ecircum/360.
	degrad=180./pi
	zmax = 9999 ! maximum depth
	nump = 9998 ! number of rays
	ideptype = 1 ! Source depths:  (1) Range, (2) Exact


!--------------------- INPUT AND SETUP -------------------------------------!
	
	
	! -------- Parse Input for TT table ------------- !
	print *, 'Enter name of output travel-time table '
	read *, ttfile
	print *,ttfile
	print *, 'Enter phase for travel time-table: P=1, S=2'
	read *, iw
	print *, iw
	print *, 'Enter table output units: 1 = (x in km, t in sec), 2 = (x in deg, t in min.)'
	read  *, itype
	print *, itype
	print *, 'Enter travel-time table depths (km):  min, max, space'
	read *, dep1,dep2,dep3
	print *, dep1,dep2,dep3
	
	print *, 'Enter travel-time table distances (del):  min, max, space'
	read *, del1,del2,del3
	print *, del1,del2,del3
	
	! ------ Parse Inputs for Velocity Model ------------!
	print *, 'Enter assumed Vp/Vs if Vs not specified in input file: '
	read  *, sfact
	print *, sfact
	print *, 'Enter min ray param (p-P, p-S) at long range (.133 = no Pn, .238 = no Sn):'
	read  *, plongcut
	print *, plongcut
	print *, 'Enter name of velocity model (for reference only)'
	read  *, vmodel
	print *, vmodel
	print *, 'Enter velo. model input format for first column (1=depth, 2=radius)'
	read  *, ideprad
	print *, ideprad
	print *, 'Enter number of lines in velocity model'
	read  *, npts
	print *, npts
	print *, 'Enter velocity model, with one line (Z, Vp, Vs) for each model depth point'
	print *, 'Note that the program uses linear interpolations between model lines,'
	print *, 'so models with constant velocity layers should include a depth point at' 
	print *, 'both the layer top and bottom.'
	do i = 1, npts
	    read *, z_inp(i), alpha_inp(i), beta_inp(i)
	    if (beta_inp(i) == 0) beta_inp(i) = alpha_inp(i)/sfact
	    write(*,'(f10.3,2f8.4)'), z_inp(i), alpha_inp(i), beta_inp(i)
	enddo
	


!------- Interpolate Velocity Model and Perform Flat-Earth Transformation ------------------!

	!----- interpolate input model, while preserving layer interfaces: 
	!      z_inp,alpha_inp,beta_inp --> z_s,alpha_s,beta_s 
	print *, 'Interpolating velocity model to source depths...'
	kk=0
	dz = dep3 ! set spacing equal to the travel time table depth spacing
	nz = floor(z_inp(npts)/dz)+1 ! number of interpolation pts
	do iz=1,nz
	 zz = (iz-1)*dz ! interpolation point
 
	 do i=1,npts-1 ! loop over velocity model layers
 
		z1=z_inp(i) ! top layer
		z2=z_inp(i+1) ! bottom layer
	
		if (z1<=zz.and.z2>zz) then ! match with interpolation pt 
	
		   ! get interpolation factor
		   if (z2==z1) then ! check for layer interface (prevent divide by zero)
				fact = 0.  
		   else  ! not an interface: interpolate
				fact=(zz-z1)/(z2-z1)
		   endif
		 
		   ! compute v at interpolation point
		   alpha0=alpha_inp(i)+fact*(alpha_inp(i+1)-alpha_inp(i))
		   beta0=beta_inp(i)+fact*(beta_inp(i+1)-beta_inp(i))
	   
			! append to interpolated velocity model (skip duplicate lines)
		   if ((kk==0).or.(z_s(kk)/=zz).or.(alpha_s(kk)/=alpha0).or.(beta_s(kk)/=beta0)) then
				kk=kk+1
				z_s(kk)=zz
				alpha_s(kk)=alpha0
				beta_s(kk)=beta0
		   endif
		 
			!check for layer points we might miss
			 do k = i+1, npts
				if (z_inp(k)-zz > dz) exit
				kk=kk+1
				z_s(kk)=z_inp(k)
				alpha_s(kk)=alpha_inp(k)
				beta_s(kk)=beta_inp(k)              
			enddo
	   
		   exit ! exit inner loop over velo model points
		end if   
	 enddo ! end loop on velocity model layers
 
	enddo ! end loop on interpolation pts

	! print out interpolated model
	nz_s = kk
	do i = 1,nz_s
	write(*,'(f10.3,2f8.4)'), z_s(i), alpha_s(i), beta_s(i)
	enddo


	!  ----- transform to flat earth -------!
	do i=1,nz_s
	  if (ideprad.eq.2) z_s(i)=erad-z_s(i) ! note erad is earth radius
	  if (z_s(i).eq.erad) then !value at center of earth is removed to avoid singularity in transformation
		nz_s = i-1
		exit
	  endif
	  call FLATTEN(z_s(i),alpha_s(i),z(i),alpha(i)) ! flat-earth transform for Vp
	  call FLATTEN(z_s(i),beta_s(i),z(i),beta(i)) ! flat-earth transform for Vs
	enddo   

	!-------- set up dummy interface at bottom
	i = nz_s+1
	z_s(i)=z_s(i-1)                  
	alpha_s(i)=alpha_s(i-1)
	beta_s(i)=beta_s(i-1)
	call FLATTEN(z_s(i),alpha_s(i),z(i),alpha(i))
	call FLATTEN(z_s(i),beta_s(i),z(i),beta(i))
	npts=i
	print *,'Depth points in model= ',npts

	! ----- compute slowness
	do i=1,npts
	 slow(i,1)=1./alpha(i)
	 if (beta(i).ne.0.) then
		slow(i,2)=1./beta(i)
	 else
		slow(i,2)=1./alpha(i)              !fluid legs are always P!
	 end if       
	enddo

	! --------- print out velocity table
	print *,'************************* Table of Model Interfaces *****', &
	'*****************'
	print *,' Depth  Top Velocities  Bot Velocities    -----Flat Earth ', &
	' Slownesses-----'
	print *,'             vp1  vs1        vp2  vs2       p1      p2  ', &
	  '      s1      s2'
	do i=2,npts
	 if (i.eq.2.or.z(i).eq.z(i-1)) then
		scr1=1./alpha(i-1)
		scr2=1./alpha(i)
		scr3=999.
		if (beta(i-1).ne.0.) scr3=1./beta(i-1)
		scr4=999.
		if (beta(i).ne.0.) scr4=1./beta(i)
		print 42,z_s(i),i-1,alpha_s(i-1),beta_s(i-1), &
		  i,alpha_s(i),beta_s(i), scr1,scr2,scr3,scr4
	42          format (f6.1,2(i5,f6.2,f5.2),2x,2f8.5,2x,2f8.5)
	 end if
	enddo

	!  setup range of source depths
	 dep2=dep2+dep3/20.
	 idep=0
	 ndep = floor((dep2-dep1)/dep3)+1
	 do idep = 1,ndep
		dep = dep1 + (idep-1)*dep3
		deptab(idep)=dep
	 enddo   

!-----------------------------------------------------------------------------------------      
      

! --------------------------------- ray tracing ------------------------------------------
!    shoot rays with different ray parameters (nump total) from the surface,
!    and keep track of the offset distance (x) and travel time (t) to different depths

      ! get number of rays to compute     
      pmin=0.
      pmax=slow(1,iw)
      print *,'pmin, pmax = ', pmin, pmax
      print *,'Number of rays to compute:'
      print *, nump   
      pstep=(pmax-pmin)/float(nump)

      np=0
200   do np = 1, nump ! ------- loop over ray parameters (p) ---------------
      
         ! current ray parameter
         p = pmin + (np-1)*pstep
         ptab(np)=p

         x=0. ! rays start at x,t = 0,0
         t=0.
         xcore=0.
         tcore=0.
         
         imth=3  !preferred v(z) interpolation method, optimal for flat-earth transform
         
         ! initialize arrays: depxcor, deptcor, depucor (size nump by ndep), which track
         ! the offset (x) and travel time (t) for different rays to different depths 
         do idep=1,ndep
            if (deptab(idep).eq.0.) then
               depxcor(np,idep)=0.
               deptcor(np,idep)=0.
               depucor(np,idep)=slow(1,iw)
            else
               depxcor(np,idep)=-999.
               deptcor(np,idep)=-999.
               depucor(np,idep)=-999.
            end if
        enddo

         do i=1,npts-1 ! ------ loop over layers (i) ----------------------------
         
             !check to see if z exceeds zmax
             if (z_s(i).ge.zmax) then                          
                deltab(np)=-999.
                tttab(np)=-999.
                go to 200
             end if

             ! layer thickness
             h=z(i+1)-z(i)							 
             if (h.eq.0.) cycle    !skip if interface
             
            ! LAYERTRACE calculates the travel time and range offset for ray tracing through a single layer.
			!  Input:   p     =  horizontal slowness
			!           h     =  layer thickness
			!           utop  =  slowness at top of layer
			!           ubot  =  slowness at bottom of layer
			!           imth  =  interpolation method
			!                    imth = 1,  v(z) = 1/sqrt(a - 2*b*z)  fastest to compute
			!                         = 2,  v(z) = a - b*z            linear gradient
			!                         = 3,  v(z) = a*exp(-b*z)        referred when Earth Flattening is applied
			!  Returns: dx    =  range offset
			!           dt    =  travel time
			!           irtr  =  return code (-1: zero thickness layer, 0: ray turned above layer, 
			!                 =    1: ray passed through layer, 2: ray turned within layer, 1 segment counted)
             call LAYERTRACE(p,h,slow(i,iw),slow(i+1,iw),imth,dx,dt,irtr) ! compute dx, dt for layer
             
             ! update x,t after tracing through layer
             x=x+dx
             t=t+dt
             
             ! exit when ray has turned
             if (irtr.eq.0.or.irtr.eq.2) exit  
         
         	! save current x,t,u for ray sampling source depths (stored in deptab)
             do idep=1,ndep
                if (abs(z_s(i+1)-deptab(idep)).lt.0.1) then
                   depxcor(np,idep)=x
                   deptcor(np,idep)=t
                   depucor(np,idep)=slow(i+1,iw)            
                end if
             enddo

!   
         enddo !------------ end loop on layers----------------------------------
         
         ! compute final (surface-to-surface) two-way offset and travel times for ray
         x=2.*x
         t=2.*t
         deltab(np)=x                   !stored in km
         tttab(np)=t                    !stored in seconds

      enddo             !---------------- end loop on ray parameter p --------------------
      print *,'Completed ray tracing loop'

!----------------------------------------------
      
! special section to get (0,0) point
         np=np+1
         ptab(np)=slow(1,iw)
         deltab(np)=0.
         tttab(np)=0.
         do idep=1,ndep
            if (deptab(idep).eq.0.) then
               depxcor(np,idep)=0.
               deptcor(np,idep)=0.
               depucor(np,idep)=slow(1,iw)
            else
               depxcor(np,idep)=-999.
               deptcor(np,idep)=-999.
               depucor(np,idep)=-999.
            end if         
         enddo

!-------------------------------------------------------------------------


!-----  Now compute T(X,Z) for first arriving rays
      
      do idep=1,ndep ! loop over source depth's (Z)
         
         
         icount=0 ! save array index
         xold=-999. ! current x
         
         ! if source is at 0 depth, go skip the upgoing ray loop
         if (deptab(idep).eq.0.) then
            i2=np
            go to 223
         end if
         
         ! loop for upgoing rays from the source
         do i=1,np                        
            x2=depxcor(i,idep)              ! offset at this source depth
            if (x2.eq.-999.) exit          
            if (x2.le.xold) exit            !stop when ray heads inward
            t2=deptcor(i,idep)
            icount=icount+1                  ! increment save index
            xsave(icount)=x2                 ! save offset from this depth to surface
            tsave(icount)=t2                 ! save travel time
            psave(icount)=-ptab(i)           ! save p as negative for upgoing from source
            usave(icount)=depucor(i,idep)    ! save slowness
            xold=x2
         enddo
         i2=i-1
         
         ! loop for downgoing rays from the source
223      continue         
         do i=i2,1,-1                  
            if (depxcor(i,idep).eq.-999.) cycle ! skip
            if (deltab(i).eq.-999.) cycle       ! skip
            x2=deltab(i)-depxcor(i,idep)    ! source-surface offset is total offset minus offset from downgoing leg 
            t2=tttab(i)-deptcor(i,idep)     ! same for source surface travel time
            icount=icount+1                 ! increment save index
            xsave(icount)=x2                ! save offset from this depth to surface
            tsave(icount)=t2                ! save p as negative for upgoing from source
            psave(icount)=ptab(i)           ! save p as negative for upgoing from source
            usave(icount)=depucor(i,idep)   ! save slowness
            xold=x2
         enddo   
         ncount=icount
         
         
         ! interpolate offsets to the desired spacing and find the first-arriving ray
         ndel = floor((del2-del1)/del3) + 1 ! number of interpolation pts
         
         do idel = 1, ndel !---------- loop over offsets
            
            deldel = del1 + (idel-1)*del3 ! current offset in km
            
            ! convert from km to degree, if desired
            del=deldel
            if (itype.eq.2) del=deldel*kmdeg
            delttab(idel)=deldel
            
            ! search for first arriving ray at this offset
            tt(idel,idep)=99999.
            do i=2,ncount
               x1=xsave(i-1)
               x2=xsave(i)
               if (x1.gt.del.or.x2.lt.del) cycle
               if (psave(i).gt.0..and.psave(i).lt.plongcut) cycle
               frac=(del-x1)/(x2-x1)
               tbest=tsave(i-1)+frac*(tsave(i)-tsave(i-1))
               if (psave(i-1).le.0..and.psave(i).le.0. .or. &
                         psave(i-1).ge.0..and.psave(i).ge.0.) then
                  pbest=psave(i-1)+frac*(psave(i)-psave(i-1))
                  ubest=usave(i-1)+frac*(usave(i)-usave(i-1)) 
               else
                  if (frac.lt.0.5) then
                     pbest=psave(i-1)
                     ubest=usave(i-1)
                  else
                     pbest=psave(i)
                     ubest=usave(i)
                  end if
               end if
              
               if (tbest.lt.tt(idel,idep)) then
                  tt(idel,idep)=tbest
                  scr1=pbest/ubest
                  if (scr1.gt.1.) then
                     scr1=1.
                  end if
                  angle=asin(scr1)*degrad
                  if (angle.lt.0.) then
                     angle=-angle
                  else
                     angle=180.-angle
                  end if
                  angang(idel,idep)=angle
                  rayray(idel,idep)=pbest
                  etaeta(idel,idep)=ubest*sqrt(1.-scr1**2)                  
                  if (angang(idel,idep).lt.90.) then
                     etaeta(idel,idep)=-etaeta(idel,idep)
                  endif
               end if
            enddo
            
            ! no ray arrivals
            if (tt(idel,idep).eq.99999.) tt(idel,idep)=0.    
            if (itype.eq.2) tt(idel,idep)=tt(idel,idep)/60.            

         enddo                                    !end loop on offsets
                  
      enddo                                        !end loop on depth

      ! fix edge cases
      if (delttab(1).eq.0.) then
         if (deptab(1).eq.0.) tt(1,1)=0.                 !set tt to zero at (0,0)
         do idep=1,ndep
            angang(1,idep)=0.                 !straight up at zero range
            etaeta(1,idep)=-abs(etaeta(1,idep))
         enddo

      end if

!-------------- Make output files -----------------

! get file name from input
      print *,'Output file name for travel-time table:'
      print *, ttfile
      open (11,file=ttfile)     
      
! write headers for all files
     
! first line: model name, iw (1/2 for P/S), pmin, pmax, np   
      write (11,407) vmodel(1:20),iw,pmin,pmax,np  
407    format ('From deptable, file= ',a20,' iw =',i2,' pmin=',f8.5, &
              ' pmax=',f8.5,' np=',i6)
     
! second line: table size ndel,ndep:
!       ndel rows (different X/DEL offsets)
!       ndep columns (different source depths)
      write (11,408) ndel,ndep
408      format (2i5)
      
!     third line: row of source depths
      write (11,409) (deptab(j),j=1,ndep)
409      format (8x,100f8.1)
      
!   fill in table:
!       first column is X/DEL of each row
!       then TT_ij = TT(x_i, z_j), where: TT_ij is the travel time of first arriving ray 
!       from a source at horizontal distance x_i and depth z_j
      do i=1,ndel
         if (itype.eq.1) then
            write (11,410) delttab(i),(tt(i,j),j=1,ndep)
         else
            write (11,413) delttab(i),(tt(i,j),j=1,ndep)
         end if

410      format (101f8.4)
413      format (f8.3,100f8.4)

      enddo
      close (11)

999   return

	
end program make_TTtable 

!================================ END PROGRAM ============================================


!================================ SUBROUTINES ============================================

!-----------------------------------------------------------------------
! INTERP finds the y3 value between y1 and y2, using the
! position of x3 relative to x1 and x2.
      subroutine INTERP(x1,x2,x3,y1,y2,y3)
      fract=(x3-x1)/(x2-x1)
      y3=y1+fract*(y2-y1)
      return
      end

!-----------------------------------------------------------------------
! FLATTEN calculates flat earth tranformation.
      subroutine FLATTEN(z_s,vel_s,z_f,vel_f)
      erad=6371.
      r=erad-z_s
      z_f=-erad*alog(r/erad)
      vel_f=vel_s*(erad/r)
      return
      end

!-----------------------------------------------------------------------
! UNFLATTEN is inverse of FLATTEN.
      subroutine UNFLATTEN(z_f,vel_f,z_s,vel_s)
      erad=6371.
      r=erad*exp(-z_f/erad)
      z_s=erad-r
      vel_s=vel_f*(r/erad)
      return
      end

!
!-----------------------------------------------------------------------
! LAYERTRACE calculates the travel time and range offset
! for ray tracing through a single layer.
!
! Input:    p     =  horizontal slowness
!           h     =  layer thickness
!           utop  =  slowness at top of layer
!           ubot  =  slowness at bottom of layer
!           imth  =  interpolation method
!                    imth = 1,  v(z) = 1/sqrt(a - 2*b*z)     fastest to compute
!                         = 2,  v(z) = a - b*z               linear gradient
!                         = 3,  v(z) = a*exp(-b*z)           preferred when Earth Flattening is applied
!
! Returns:  dx    =  range offset
!           dt    =  travel time
!           irtr  =  return code
!                 = -1, zero thickness layer
!                 =  0,  ray turned above layer
!                 =  1,  ray passed through layer
!                 =  2,  ray turned within layer, 1 segment counted
!
! Note:  This version does calculation in double precision,
!        but all i/o is still single precision
!
      subroutine LAYERTRACE(p1,h1,utop1,ubot1,imth,dx1,dt1,irtr)
      implicit real*8 (a-h,o-z)
      real*4 p1,h1,utop1,ubot1,dx1,dt1
      
      ! double precision
      p=dble(p1)
      h=dble(h1)
      utop=dble(utop1)
      ubot=dble(ubot1)

      !check for zero thickness layer
      if (h.eq.0.) then                  
         dx1=0.
         dt1=0.
         irtr=-1
         return         
      end if

      ! slowness of top layer
      u=utop
      
      !check for complex vertical slowness: ray turned above layer
      y=u-p
      if (y.le.0.) then                       
         dx1=0.
         dt1=0.
         irtr=0
         return
      end if


	  ! qs = vertical slowness: sqrt(u^2-p^2)
      q=y*(u+p)
      qs=dsqrt(q)

      ! special function needed for integral at top of layer
      if (imth.eq.2) then
         y=u+qs
         if (p.ne.0.) y=y/p
         qr=dlog(y)
      else if (imth.eq.3) then ! flat-earth
         qr=atan2(qs,p)
      end if      


      ! b factor (ray tracing integral constant)
      if (imth.eq.1) then
          b=-(utop**2-ubot**2)/(2.*h)
      else if (imth.eq.2) then
          vtop=1./utop
          vbot=1./ubot
          b=-(vtop-vbot)/h
      else                     
          b=-dlog(ubot/utop)/h   ! flat earth
      end if  
!
	  !constant velocity layer
      if (b.eq.0.) then                         
         b=1./h
         etau=qs
         ex=p/qs
         irtr=1
         go to 160
      end if

	! ray tracing integral at upper limit, 1/b factor omitted until end
      if (imth.eq.1) then
         etau=-q*qs/3.
         ex=-qs*p
      else if (imth.eq.2) then
         ex=qs/u                       !*** - in some versions (wrongly)
         etau=qr-ex
         if (p.ne.0.) ex=ex/p
      else
         etau=qs-p*qr                  ! flat-earth
         ex=qr
      end if

	 ! check lower limit to see if we have turning point
      u=ubot
      if (u.le.p) then                                !if turning point,
         irtr=2                                       ! then no contribution
         go to 160                                    !from bottom point
      end if
      
      ! no turning point: ray passes through 
      irtr=1
      q=(u-p)*(u+p)
      qs=dsqrt(q)
!
      if (imth.eq.1) then
         etau=etau+q*qs/3.
         ex=ex+qs*p
      else if (imth.eq.2) then
         y=u+qs
         z=qs/u
         etau=etau+z
         if (p.ne.0.) then
            y=y/p
            z=z/p
         end if
         qr=dlog(y)
         etau=etau-qr
         ex=ex-z
      else
         qr=atan2(qs,p)
         etau=etau-qs+p*qr
         ex=ex-qr
      end if      


      ! ray tracing equations to get dt, dx
160   dx=ex/b         ! horizontal offset in km
      dtau=etau/b     ! delay time
      dt=dtau+p*dx    ! convert delay time to travel time

      ! back to single precision at the end
      dx1=sngl(dx)
      dt1=sngl(dt)
      return
      end

!-----------------------------------------------------------------------
