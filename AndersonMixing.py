#Anderson Mixing for a homopolymer brush of wormlike chains. Generates rho and rhoz
#Which are the density as a function of orientation/position and only position
#respectively. phi_new is the target concentration, w is the field. I have
#phi_new normalized so that it's total concentation is 1 (after integrating over z)

#Load all the packages I might need
from __future__ import division # must be first
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
from datetime import datetime
startTime = datetime.now()
print(startTime)
## Define Basic Constants ##
kappa = 1/32.0                      # kappa is the stiffness of the polymer
kappainv = kappa**(-1.0)            # 1/kappa
rho_0 = 1                           # polymer density
N = 1                               # N is the number of monomers
b_0 = 1                             # Average distance between monomers
l_c = b_0*N                         # contour length
l_p = b_0*kappa                     # persistence length
R_0 = (2*l_p*l_c)**0.5              # approximate radius of gyration assuming flexible limit
sigma = 1                           # Grafting density
rho_0 = 1                           # polymer density
constant = (8*(sigma*N/(b_0*rho_0))*l_p/(np.pi**2.0*l_c))**(1/3.0)*l_c/R_0
nu = 1**3*constant**(-3.0)          # excluded volume parameter (if x**3*constant**(-3.0) then L/R_0 = x)
Lambda_0 = nu*sigma*N/(b_0*rho_0)   # excluded volume interation parameter in form in thesis
L_class = (8*Lambda_0*l_p/(np.pi**2.0*l_c))**(1/3.0)*l_c # Classical brush height
#############################
b = b_0 * N # Contour length
c = ( N * kappainv) / (2 * kappa )  # Constant used in numerical method
trunc = 10                          # Cut off of Legendre polynomials
#Load Wigner 3j symbols: Gamma is a special case of the Wigner-3j symbol.
# It is a nxn array with element psi[ [l1], [l2], [l3]] = (2l1+1) * gamma^2
psi1 = np.load('psi.npy')
LP1 = np.load('LP.npy')                             # Legendre polynomials Pl(uz)
negLP1 = np.load('negLP.npy')                       # Pl(-uz)
L1 = np.load('L.npy')                               # diagonal matrix with entries l(l+1)
Lhalf1 = np.load('Lhalf.npy')                       # diagonal matrix with entries 1/(l+1/2)
invLhalfalt1 = np.load('invLhalfalt.npy')           # vector with entries (-1)^(l)(l+1/2)
alternatingsign1 = np.load('alternatingsign.npy')   # Vector with entries (-1)^l
# Cut off at truncation trunc
psi = psi1[:(trunc+1),:(trunc+1),:(trunc+1)]
LP = LP1[:(trunc+1),:(201)]
negLP = negLP1[:(trunc+1),:(201)]
L = L1[:(trunc+1),:(trunc+1)]
Lhalf = Lhalf1[:(trunc+1),:(trunc+1)]
invLhalfalt = invLhalfalt1[:(trunc+1)]
alternatingsign = alternatingsign1[:(trunc+1),:(trunc+1)]
# Define step size
# Increase Lz and Ls to increase discretization
# Increase only dz to increase box size
Lz = 2*(10**(2)) + 1                            # z varies from 0 to 1
Ls = 8*(10**(2)) + 1                           # s varies from 0 to 1
ds = 1/(Ls-1)                                   # s is contour length
dz = 1/(Lz-1)*1                                 # z is position
#Define coordinates
l = np.linspace(0,trunc, num = trunc+1)         # Legendre poly index
z = np.linspace(0,Lz-1, num = Lz)*dz            # z vector for plotting
s = np.linspace(0,Ls-1, num = Ls)*ds            # s vector for plotting
uz = np.linspace(1,-1,201)                      # Orientation (uz= cos(\theta))
theta = np.arccos(uz)
phi = np.linspace(0,2*np.pi,201)
sintheta = np.sin(theta)
costheta = uz
cosphi = np.cos(phi)
#Intial conditions
q0 = np.zeros((trunc+1,Lz,Ls))              #Initialize the propagater
#Intial conditions
q0 = np.zeros((trunc+1,Lz,Ls))              #Initialize the propagater
#Forward propagator initial condition
q = np.copy(q0)
q[0,:,0] = np.exp(-(32*(z-0.01))**2)
qzneg1 = np.zeros(trunc+1)
qzneg1[0] = 0
#q[:,0,0] = np.einsum('ij,j->i',Lhalf,LP[:,0])
qdagger = np.copy(q0)
qdagger[0,:,0] = 1#/np.trapezoid(np.ones(Lz),z)
#Field initial condition
#Target Concentration
#cosgamma is theta by theta prime by phi array
cosgamma = np.einsum('i,j->ij',uz,uz)[...,None] + np.einsum('ij,k->ijk',np.einsum('i,j->ij',(np.abs(1-uz**2))**(0.5),(np.abs(1-uz**2))**(0.5)),cosphi)
ucrossuprime = (np.abs(1-cosgamma**2))**(0.5)
intucrossuprime = np.trapezoid(ucrossuprime,phi,axis = 2)
phi_newfactor1 = np.einsum('ij,jk->ijk',LP,intucrossuprime) #???
intphi_newfactor1 = np.trapezoid(phi_newfactor1,uz,axis = 1)
phi_newfactor2 = np.einsum('lk,ik->ilk',np.einsum('ij,jk->ik',Lhalf,LP),intphi_newfactor1)
intphi_newfactor2 = np.trapezoid(phi_newfactor2,uz,axis = 2)
w = np.zeros((trunc+1,Lz))                  # Initialize the Field
dzw = np.zeros((trunc+1,Lz))                # Field gradient

#This Function Calculates the Concentration
def Concentration(b,c,psi,LP,negLP,L,Lhalf,invLhalfalt,alternatingsign,Lz,Ls,ds,dz,q,w,dzw):
    #Calculates the Concentration rho
    #w[0,0:(Lz-1)/10] = 100*np.exp(-(z[0:(Lz-1)/10])**2*5)
    #w[0,:] += -50
    for k in  range(0,Lz):
        #empty B.C. at z = 0
        if k == 0:
            dzw[:,k] = (1/(dz)) * (w[:,k+1]-w[:,k+1])
        elif k+1 == Lz:
            dzw[:,k] = 0*(1/(dz)) * (w[:,k-1]-w[:,k-1])
        else:
            dzw[:,k] = (1/(2 * dz)) * (w[:,k+1]-w[:,k-1])
    #Constants that go into my final set of constants
    psidotw = np.tensordot( psi , (w), axes = ((0),(0)) ) #I will need this
    psidotdzw = np.tensordot( psi , (dzw), axes = ((0),(0)) )
    #Define my final set of Constants for the forward propagating solution
    #They are l x l x Lz arrays
    A1 =  c * L / 2
    A2 =   (1 / 2) * psidotw
    A3 =  ( np.eye(trunc+1) ) / ds
    A = (A1[...,None] + A2 + A3[...,None])
    # ainv is Lz x l x l
    AinvT = np.linalg.inv(np.transpose(A))
    Ainv = np.transpose(AinvT)
    A = np.eye(trunc+1)[...,None] + np.zeros((trunc+1,trunc+1,Lz))

    B1 = (ds/(2 * (dz)**2)) * b**2 * np.dot(psi[1,:,:],psi[1,:,:])
    B = B1[...,None] + np.zeros((trunc+1,trunc+1,Lz))
    B = np.einsum('ijk,jlk->ilk',Ainv,B)

    C1 = (1/2) * c * np.einsum('ij,jk->ik',psi[1,:,:],L)
    C2 = (1/2) * np.einsum('ik,kjl->ijl',psi[1,:,:],psidotw)
    C3 = (1/ds) * psi[1,:,:]
    C = (ds/(2 *dz)) * b * (C1[...,None] + C2 - C3[...,None])
    C = np.einsum('ijk,jlk->ilk',Ainv,C)

    D1 = (ds/2) * b * np.einsum('ik,kjl->ijl',psi[1,:,:],psidotdzw)
    D2 = c * L
    D3 = psidotw
    D = D1 - D2[...,None] - D3
    D = np.einsum('ijk,jlk->ilk',Ainv,D)

    # Calculate forward propogating solution
    for i in range(0, Ls-1):
        #0 condition at z = 0
        q1 =  np.tensordot( (B[:,:,0] - C[:,:,0] ) , qzneg1, axes = ((1),(0))  )
        q2 =  np.tensordot( (A[:,:,0] + D[:,:,0]- 2 * B[:,:,0]) , q[ :, 0, i ], axes = ((1),(0))  )
        q3 =  np.tensordot( (B[:,:,1] + C[:,:,1]) , q[ :, 1, i ], axes = ((1),(0)) )
        q4 =  (q1 + q2 + q3)
        q[ :, 0, i+1 ] = np.copy(q4)
        # z components
        q1 = np.einsum('ijk,jk->ik',(B[:,:,:] - C[:,:,:] ),q[ :, :, i ],optimize = True) # z-1 element
        q2 = np.einsum('ijk,jk->ik',(A[:,:,:] + D[:,:,:]- 2 * B[:,:,:]),q[ :, :, i ],optimize = True) # z element
        q3 = np.einsum('ijk,jk->ik',(B[:,:,:] + C[:,:,:]),q[ :, :, i ],optimize = True) # z+1
        q4 = q1[:,0:-2] + q2[:,1:-1] + q3[:,2::]
        q[ :, 1:-1, i+1 ] = np.copy(q4)
        # for j in range(1, Lz-1):
            # q1 =  np.tensordot( (B[:,:,j-1] - C[:,:,j-1] ) , q[ :, j-1, i ], axes = ((1),(0))  )
            # q2 =  np.tensordot( (A[:,:,j] + D[:,:,j]- 2 * B[:,:,j]) , q[ :, j, i ], axes = ((1),(0))  )
            # q3 =  np.tensordot( (B[:,:,j+1] + C[:,:,j+1]) , q[ :, j+1, i ], axes = ((1),(0)) )
            # q4 =  (q1 + q2 + q3)
            # q[ :, j, i+1 ] = np.copy(q4)
        #reflecting condition at z = Lz
        q1 =  np.tensordot( (B[:,:,Lz-2] - C[:,:,Lz-2] ) , q[ :, Lz-2, i ], axes = ((1),(0))  )
        q2 =  np.tensordot( (A[:,:,Lz-1] + D[:,:,Lz-1]- 2 * B[:,:,Lz-1]) , q[ :, Lz-1, i ], axes = ((1),(0))  )
        q3 =  np.tensordot( (B[:,:,Lz-1] + C[:,:,Lz-1]) , q[ :, Lz-1, i ], axes = ((1),(0)) )
        q4 =  (q1 + q2 + q3)
        q[ :, Lz-1, i+1 ] = np.copy(q4)
    #qdagger
    for i in range(0, Ls-1):
        #0 condition at z = 0
        qd1 =  np.tensordot( (B[:,:,0] - C[:,:,0] ) , qzneg1, axes = ((1),(0))  )
        qd2 =  np.tensordot( (A[:,:,0] + D[:,:,0]- 2 * B[:,:,0]) , qdagger[ :, 0, i ], axes = ((1),(0))  )
        qd3 =  np.tensordot( (B[:,:,1] + C[:,:,1]) , qdagger[ :, 1, i ], axes = ((1),(0)) )
        qd4 =  (qd1 + qd2 + qd3)
        qdagger[ :, 0, i+1 ] = qd4
        # z components
        qd1 = np.einsum('ijk,jk->ik',(B[:,:,:] - C[:,:,:] ),qdagger[ :, :, i ],optimize = True) # z-1 element
        qd2 = np.einsum('ijk,jk->ik',(A[:,:,:] + D[:,:,:]- 2 * B[:,:,:]),qdagger[ :, :, i ],optimize = True) # z element
        qd3 = np.einsum('ijk,jk->ik',(B[:,:,:] + C[:,:,:]),qdagger[ :, :, i ],optimize = True) # z+1
        qd4 = qd1[:,0:-2] + qd2[:,1:-1] + qd3[:,2::]
        qdagger[ :, 1:-1, i+1 ] = np.copy(qd4)
        # for j in range(1, Lz-1):
            # qd1 =  np.tensordot( (B[:,:,j-1] - C[:,:,j-1] ) , qdagger[ :, j-1, i ], axes = ((1),(0))  )
            # qd2 =  np.tensordot( (A[:,:,j] + D[:,:,j]- 2 * B[:,:,j]) , qdagger[ :, j, i ], axes = ((1),(0))  )
            # qd3 =  np.tensordot( (B[:,:,j+1] + C[:,:,j+1]) , qdagger[ :, j+1, i ], axes = ((1),(0)) )
            # qd4 =  (qd1 + qd2 + qd3)
            # qdagger[ :, j, i+1 ] = np.copy(qd4)
        #reflecting condition at z = Lz
        qd1 =  np.tensordot( (B[:,:,Lz-2] - C[:,:,Lz-2] ) , qdagger[ :, Lz-2, i ], axes = ((1),(0))  )
        qd2 =  np.tensordot( (A[:,:,Lz-1] + D[:,:,Lz-1]- 2 * B[:,:,Lz-1]) , qdagger[ :, Lz-1, i ], axes = ((1),(0))  )
        qd3 =  np.tensordot( (B[:,:,Lz-2] + C[:,:,Lz-1]) , qdagger[ :, Lz-1, i ], axes = ((1),(0)) )
        qd4 =  (qd1 + qd2 + qd3)
        qdagger[ :, Lz-1, i+1 ] = np.copy(qd4)
    #sum over Legendre's
    #qsz[s,z,theta]
    qsz = np.dot( ( np.transpose(q) ), (LP[:(trunc+1),:]) )
    #Calculate partition function
    #using q(uz)
    qdsz1 = (np.dot( ( np.transpose(qdagger) ), (negLP[:(trunc+1),:]) ) )
    qdsz = np.copy(qdsz1[::-1,:,:])
    qqdsz = qsz * qdsz
    #Using ql's
    qd = np.copy(qdagger[:,:,::-1])
    qqdl = q * qd
    qqd = np.dot( np.transpose(qqdl), (invLhalfalt) )
    Q = np.trapezoid(qqd,z,axis = 1)
    #Calculate density, rhol(l,z), rho(z,uz)
    rhoz = np.trapezoid(qqd,s,axis =0)/Q[-1]             #rhoz has been integrated over orientations
    rho = np.trapezoid(qqdsz,s,axis = 0)/Q[-1]           #rho has orientation
    #rhoend = qqdsz[0,:,:]/Q[0]
    #rhoendz = qqd[0,:]/Q[0]
    #Make my chains twice as long
    #rho = (qsz[Ls-1,:,:]*qsz[Ls-1,:,::-1])/Q[0]
    #qql = q * q
    #qq = np.dot( np.transpose(qql), (invLhalfalt) )
    #rhoz = qq[Ls-1,:]/Q[0]
    rhoend = qqdsz[Ls-1,:,:]/Q[-1]
    rhoendz = qqd[Ls-1,:]/Q[-1]
    qalternatingsign = np.einsum('ij,jmn->imn',alternatingsign,qd[:,:,:])
    psi2 = np.einsum('ijk,imn->jkmn',psi,q[:,:,:])
    rhols = np.einsum('jkmn,kmn->jmn',psi2,qalternatingsign)
    rhol = np.trapezoid(rhols,s,axis = 2)/Q[-1]
    return Q, q, qdagger, rhol, rho, rhoz,rhoend, rhoendz
#rho, rhoz = Concentration(b,c,psi,LP,negLP,L,Lhalf,invLhalfalt,alternatingsign,Lz,Ls,ds,dz,q,w,dzw)
def Anderson(w,nu,sigma):
    DIM = 6
    DEV = np.zeros((DIM,trunc+1,Lz))
    DEV0 = np.zeros((DIM,trunc+1,Lz))
    DDEV = np.zeros((DIM,trunc+1,Lz))
    WIN = np.zeros((DIM,trunc+1,Lz))
    U = np.zeros((DIM,DIM))
    V = np.zeros(DIM)
    C = np.zeros(DIM)
    rho = np.zeros((200,Lz))
    rhoz = np.zeros(Lz)
    rhol = np.zeros((trunc+1,Lz))
    err = 1.0
    #errorvector = np.zeros(1000)
    k = 1
    while k < 10000 and err > 10**(-3):
        w2 = np.copy(w[:,:])
        Q, q1, qdagger1, rhol, rho, rhoz, rhoend, rhoendz = Concentration(b,c,psi,LP,negLP,L,Lhalf,invLhalfalt,alternatingsign,Lz,Ls,ds,dz,q,w,dzw)
        #wl_new = ((N**2*4*kappa*sigma)/(2*np.pi))*np.einsum('ij,ik->kj',rhol,intphi_newfactor2)
        wl_new = nu*sigma*np.einsum('ij,ik->kj',rhol,intphi_newfactor2)
        DEV[0,:,:] = wl_new[:,:] - w2[:,:]
        DEV0 = np.ones((DIM,trunc+1,Lz))*DEV[0,:,:]#????
        S1 = 0
        S2 = 0
        WIN[0,:] = w2[:,:]
        WIN0 = np.ones((DIM,trunc+1,Lz))*WIN[0,:,:] #???
        S1 += np.einsum('ij,ij->',DEV[0,:,:],DEV[0,:,:])
        S2 += np.einsum('ij,ij->',w2[:,:],w2[:,:])
        err = ((S1/(Lz-1))**(0.5))/np.amax(wl_new)
        #errorvector[k] = err
        print('The error is' ,str(err*100),"percent \n")
        lambda1 = 1 - 0.999**k
        if k < DIM:
            histories = k - 1
        else:
            histories = DIM - 1
        if k < 3:
            w2[:,:] += lambda1*DEV[0,:,:]
        else:
            #for m in range(0,histories+1):
            V[0:histories] = 0.0
            DDEV[0:histories,:,:] = DEV0[0:histories,:,:] - DEV[0:histories,:,:]
            V[0:histories] += np.einsum('ijk,ijk->i',DDEV[0:histories,:,:],DEV0[0:histories,:,:])
            S1 = 0
            #triangle = np.tri(histories)
            #DDEVtri = np.einsum('ij,jk->ik',triangle,DDEV) #???
            #S1 += np.einsum('ij,ij->',DDEV,DDEVtri)
            U = np.einsum('ijk,ljk->il', DDEV[0:histories,:,:],DDEV[0:histories,:,:])
            #print(U)
            C[1:histories] = np.linalg.lstsq(U[1:histories,1:histories],V[1:histories])[0] #Changed to 1:hist???
            w2[:,:] = WIN[0,:,:] + lambda1*DEV[0,:,:]
            w2[:,:] += np.einsum('i,ijk -> jk',C[1:histories], ( WIN[1:histories,:,:] \
            + lambda1*DEV[1:histories,:,:] ) - ( WIN0[1:histories,:,:] + lambda1*DEV0[1:histories,:,:] ) )
            #print(np.einsum('ij,j->i',U[1:histories,1:histories],C[1:histories])-V[1:histories])
            #Need n for something
        print(k)
        n=1+(k-1)%(DIM-1)
        print(n)
        DEV[n,:,:] = DEV[0,:,:]
        WIN[n,:,:] = WIN[0,:,:]
        w[:,:] = w2
        #print(k)
        print(datetime.now() - startTime)
        k += 1
    return w
W = Anderson(w,nu,sigma)
Q, q, qdagger, rhol, rho, rhoz, rhoend, rhoendz = Concentration(b,c,psi,LP,negLP,L,Lhalf,invLhalfalt,alternatingsign,Lz,Ls,ds,dz,q,w,dzw)
#############
wl_new = ((N**2*4*kappa*sigma)/(2*np.pi))*np.einsum('ij,ik->kj',rhol,intphi_newfactor2)
wzuz = np.einsum('ij,ik->jk',W,LP)
wz = np.trapezoid(wzuz,uz,axis = 1)
wzuz_new = np.einsum('ij,ik->jk',wl_new,LP)
wz_new = np.trapezoid(wzuz_new,uz,axis = 1)
##############

rho_SST = (3*R_0/(2.0*L_class**3.0))*(L_class**2.0-z**2.0) # Classical brush profile from strong-stretching-theory

index = np.argmin(np.abs(z-L_class))
rho_SST = rho_SST

plt.figure(figsize=(8, 6))
plt.plot(z/L_class,L_class*rhoz,'k',linewidth = 2.0)
plt.plot(z/L_class,L_class*rho_SST/R_0,'r:',linewidth = 2.0)

plt.ylabel('$\\phi(z)/\\sqrt{2\\ell_{\\rm p}\\ell_{\\rm c}}$',fontsize = 36)
plt.xlabel('$z/L$',fontsize = 36)
plt.ylim(0,2)
plt.xlim(0,2)
plt.xticks([0,1,2],fontsize = 24)
plt.yticks([0,1,2],fontsize = 24)
plt.tight_layout()
plt.show()