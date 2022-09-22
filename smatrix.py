import numpy as np
import warnings
from scipy import linalg as la

class DecompositionWarning(Warning):
    def __init__(self,message):
        self.message = message
        return None

    def __str__(self):
        return self.message

class AccuracyWarning(Warning):
    def __init__(self,message):
        self.message = message
        return None

    def __str__(self):
        return self.message

class ScatteringMatrix(object):
    """
    Different numbers of left and right moving modes
    are possible
    """
    def __init__(self,n,m,**kwargs):
        """
        generates scattering matrix of size (n+m) x (n+m)
        n right-moving channels
        m left-moving channels
        """
        self.n = n
        self.m = m
        self.tol = 1e-6

        # cutoff for polar decomposition - more comments below
        self.cutoff = 27

        # If unitary_correction is True, we will always pick the closest
        # unitary matrix after each construction step
        self.unitary_correction = False

        self.t  = np.eye(n)
        self.tp = np.eye(m)

        self.r  = np.zeros((m,n))
        self.rp = np.zeros((n,m))

        for key in kwargs:
            if key=='t_matrix':
                self.from_transfer(kwargs[key])
            else:
                setattr(self,key,kwargs[key])

        if self.unitary_correction:
            self.closest_unitary()

        if not self.unitary():
            message = 'Scattering matrix not unitary'
            raise ValueError(message)

        return None

    def __eq__(self,other):
        result = []
        for key in self.__dict__.keys():
            if type(getattr(self,key))==np.array:
                tmp = np.linalg.norm(getattr(self,key) - getattr(other,key))<1e-10
            else:
                tmp = getattr(self,key)==getattr(other,key)
            if type(tmp)!=bool:
                result.append(np.all(tmp))
            else:
                result.append(tmp)
        return np.all(result)

    def from_transfer(self,t_matrix):
        N = t_matrix.shape[0]
        if N!=self.n+self.m or t_matrix.shape[1]!=N:
            # wrong dimensions
            raise ValueError('t_matrix has wrong dimensions')

        self.t = np.linalg.inv(t_matrix[:self.n,:self.n].T.conjugate() )
        self.tp= np.linalg.inv(t_matrix[self.n:,self.n:])
        self.r =-np.dot(self.tp,t_matrix[self.n:,:self.n])
        self.rp= np.dot(t_matrix[:self.n,self.n:],self.tp)

        if self.unitary_correction:
            self.closest_unitary()
        
        return None

    def closest_unitary(self):
        s_old = np.bmat([[self.r,self.tp],[self.t,self.rp]])
        u,_,vH = la.svd( s_old )
        snew = u.dot(vH)
        self.r = snew[:self.m,:self.n]
        self.tp= snew[:self.m,self.n:]
        self.t = snew[self.m:,:self.n]
        self.rp= snew[self.m:,self.n:]
        return None

    def conductance(self):
        G = np.real( np.einsum('nm,mn',self.t.T.conj() ,self.t ))
        Gp= np.real( np.einsum('nm,mn',self.tp.T.conj(),self.tp))
        return G,Gp

    def tsquared(self):
        return np.real(np.einsum('nm,mn->n',self.t.T.conj(),self.t))

    def unitary(self):
        """
        Checks if scattering matrix is unitary
        S*S = [[  r*r +  t*t,  r*tp + t*rp ],
               [ tp*r + rp*t, tp*tp +rp*rp ]] = 1
        """
        u = []
        u.append( (self.r.T.conj() ).dot(self.r ) + (self.t.T.conj() ).dot(self.t )-np.eye(self.n))
        u.append( (self.r.T.conj() ).dot(self.tp) + (self.t.T.conj() ).dot(self.rp) )
        u.append( (self.tp.T.conj()).dot(self.r ) + (self.rp.T.conj()).dot(self.t ) )
        u.append( (self.tp.T.conj()).dot(self.tp) + (self.rp.T.conj()).dot(self.rp)-np.eye(self.m))

        unitary = [np.linalg.norm(u0,ord=np.inf)<self.tol for u0 in u]
        return np.all(unitary)

    def polar_decomposition(self):
        """
        Polar decomposition of scattering matrix according to
        Eq. (24) in Beenakker RMP 69, 731-808
        """
        # Singular value decomposition of scattering matrix
        q0 = la.inv(self.t).dot(self.rp)
        q1 =-self.r.dot(la.inv(self.t))

        upH,sinh0,vp = la.svd(q0)
        u,sinh0,vH = la.svd(q1)


        # When we compute singular values via sinh, small values of xn are
        # inaccurate; we introduce self.cutoff=27 (might need to be changed,
        # depending on the circumstances), and computes xn<cutoff differently
        xn = np.arcsinh(sinh0)
        M0 = np.sum(xn>self.cutoff)

        # the inaccurate eigenvalues need to be fixed by another trafo
        a0,Tn,b0 = np.linalg.svd( (vH.dot(self.t.dot(upH)) )[M0:,M0:] )
        a1,_,b1  = np.linalg.svd( (u.T.conj().dot(self.tp.dot(vp.T.conj())))[M0:,M0:] )
        q0 = [ a0[:,::-1],a1[:,::-1],b0[::-1],b1[::-1] ]

        # values below cutoff from Tn
        xn[M0:] = np.arccosh(1/Tn[::-1])

        # only the inaccurate eigenvalues need to be transformed
        q0_tmp = [np.zeros_like(self.r) for _ in range(4)]
        for l,q1_tmp in enumerate(q0_tmp):
            q1_tmp[:M0,:M0]+= np.eye(M0)
            q1_tmp[M0:,M0:]+= q0[l]

        v = vH.T.conj().dot(q0_tmp[0])
        u = u.dot(q0_tmp[1])
        up= q0_tmp[2].dot(upH.T.conj())
        vp =q0_tmp[3].dot(vp)

        # Fix the signs
        r_tmp =-u.T.conj().dot( self.r.dot(  up.T.conj() ) )
        rp_tmp= v.T.conj().dot( self.rp.dot( vp.T.conj() ) )

        phi_r = np.angle(np.diag(r_tmp))
        phi_rp= np.angle(np.diag(rp_tmp))

        u = u.dot(np.diag( np.exp(1j*phi_r) ))
        vp= np.diag( np.exp(1j*phi_rp) ).dot( vp )

        tp_tmp= u.T.conj().dot( self.tp.dot( vp.T.conj() ) )
        t_tmp = v.T.conj().dot( self.t.dot(  up.T.conj() ) )
        r_tmp =-u.T.conj().dot( self.r.dot(  up.T.conj() ) )
        rp_tmp= v.T.conj().dot( self.rp.dot( vp.T.conj() ) )

        # Test: it v.H r' v'.H diagonal? And are its singular values
        # correctly related to singular values of t,t'?
        norm = []
        norm.append( np.linalg.norm(r_tmp -np.diag(np.tanh(xn))) )
        norm.append( np.linalg.norm(rp_tmp-np.diag(np.tanh(xn))) )
        norm.append( np.linalg.norm(t_tmp -np.diag(1/np.cosh(xn))) )
        norm.append( np.linalg.norm(tp_tmp-np.diag(1/np.cosh(xn))) )
        
        if np.linalg.norm(norm)>self.tol:
            message = 'Singular values of r and t not related. Problem most lik'
            message+= 'ely due to degenerate singular values. This is hopefull'
            message+= 'y inconsequential, but may in some cases result in the w'
            message+= 'rong sign of determinant. Error='+repr(np.linalg.norm(norm))
            error = DecompositionWarning(message)
            warnings.warn(error)

        return xn,u,up,v,vp

    def is_antisymmetric(self):
        """
        If there is a time reversal symmetry T with T^2 = -1, then the
        scattering matrix can be written in a basis such that S^T = -S.
        Check if this is true. Like the unitary routine, the comparison is
        with the largest absolute value.
        """
        u = []
        u.append( self.r - self.r.T  )
        u.append( self.rp- self.rp.T )
        u.append( self.tp- self.t.T  )
        u.append( self.rp- self.rp.T )

        antisymmetric = [np.linalg.norm(u0,ord=np.inf)<self.tol for u0 in u]
        return np.all( antisymmetric )

    def __mul__(self,other):
        new = ScatteringMatrix(self.n,self.m)
        new.unitary_correction = self.unitary_correction and other.unitary_correction
        new.tol = other.tol

        if self.n!=other.n or self.m!=other.m:
            raise ValueError("Number of modes is incompatible")
        if not self.unitary():
            raise ValueError("Matrix 1 not unitary")
        if not other.unitary():
            raise ValueError("Matrix 2 not unitary")

        rpr_inv = np.eye(self.n)-self.rp.dot(other.r)
        rpr_inv = np.linalg.inv(rpr_inv)
        rpr_inv = rpr_inv.dot(self.t)

        rrp_inv = np.eye(self.m)-other.r.dot(self.rp)
        rrp_inv = np.linalg.inv(rrp_inv)
        rrp_inv = rrp_inv.dot(other.tp)

        new.t = other.t.dot(rpr_inv)
        new.tp= self.tp.dot(rrp_inv)
        new.r = self.r  + self.tp.dot(other.r.dot(rpr_inv))
        new.rp= other.rp+ other.t.dot(self.rp.dot(rrp_inv))

        if new.unitary_correction:
            new.closest_unitary()
        if not new.unitary():
            raise ValueError("New matrix not unitary")

        return new

    def fano_factor(self):
        """
        Return the Fano factor given by
            F = Tr[ t.t† (1-t.t†) ]/Tr[t.t†]
        where t† is the Hermitian conjugate of t
        """
        ttH = self.t.dot(self.t.T.conj())
        fano_factor = np.trace(ttH.dot(np.eye(self.n)-ttH))/np.trace(ttH)
        return fano_factor

    def to_transfer(self):
        tp_inv = np.linalg.inv(self.tp)
        t00 = np.linalg.inv(self.t.T.conj())
        t01 = self.rp.dot(tp_inv)
        t10 =-tp_inv.dot(self.r)
        t11 = tp_inv
        return np.bmat([[t00,t01],[t10,t11]])

class DiagonalScatteringMatrix(object):
    """
    For diagonal problems, e.g., translationally invariant problems
    """
    def __init__(self,n,sps=1,**kwargs):
        """
        n channels
        each channel sps x sps
        """
        self.n = n
        self.sps = sps
        self.tol = 1e-4

        self.t  = np.zeros((n//sps,sps,sps),complex)
        self.tp = np.zeros((n//sps,sps,sps),complex)
        self.r  = np.zeros((n//sps,sps,sps),complex)
        self.rp = np.zeros((n//sps,sps,sps),complex)
        # identity
        self.t[:] = np.eye(sps)[None,:,:]
        self.tp[:] = np.eye(sps)[None,:,:]

        for key in kwargs:
            if key=='t_matrix':
                N = kwargs[key].shape[0]
                self.n = N//2
                t = []
                t.append(kwargs[key][:n,:n].reshape(sps,n//sps,sps,n//sps))
                t.append(kwargs[key][:n,n:].reshape(sps,n//sps,sps,n//sps))
                t.append(kwargs[key][n:,:n].reshape(sps,n//sps,sps,n//sps))
                t.append(kwargs[key][n:,n:].reshape(sps,n//sps,sps,n//sps))
                for t0 in t:
                    t0 = np.diagonal(np.transpose(t0,axes=(1,3,0,2)))
                    t0 = np.transpose(t0,axes=(2,0,1))
                self.t = np.conjugate(np.transpose(np.linalg.inv(t[0]),axes=(0,2,1)))
                self.tp= np.linalg.inv(t[3])
                self.r =-np.einsum('qij,qjk->qik',self.tp,t[2])
                self.rp= np.einsum('qij,qjk->qik',t[1],self.tp)

            else:
                setattr(self,key,kwargs[key])

        if not self.unitary():
            message = 'Scattering matrix not unitary'
            raise ValueError(message)

        return None


    def conductance(self):
        return np.sum( np.absolute(self.t)**2)

    def t_squared(self):
        return np.sum(np.absolute(self.t)**2,axis=(1,2))
    
    def unitary(self):
        """
        Checks if scattering matrix is unitary
        S*S = [[  r*r +  t*t,  r*tp + t*rp ],
               [ tp*r + rp*t, tp*tp +rp*rp ]] = 1
        """
        u = []
        u0 = np.einsum('qij,qik->qjk',self.r.conj(),self.r)
        u0+= np.einsum('qij,qik->qjk',self.t.conj(),self.t)
        u0+=-np.eye(self.sps)[None,:,:]
        u1 = np.einsum('qij,qik->qjk',self.r.conj(), self.tp)
        u1+= np.einsum('qij,qik->qjk',self.t.conj(), self.rp)
        u2 = np.einsum('qij,qik->qjk',self.tp.conj(),self.r )
        u2+= np.einsum('qij,qik->qjk',self.rp.conj(),self.t )
        u3 = np.einsum('qij,qik->qjk',self.rp.conj(),self.rp)
        u3+= np.einsum('qij,qik->qjk',self.tp.conj(),self.tp)
        u3+=-np.eye(self.sps)[None,:,:]

        u = [u0,u1,u2,u3]

        unitary = [np.linalg.norm(u0)<self.tol for u0 in u]
        return np.all( unitary )

    def is_antisymmetric(self):
        """
        If there is a time reversal symmetry T with T^2 = -1, then the
        scattering matrix can be written in a basis such that S^T = -S.
    	Check if this is true. Like the unitary routine, the comparison is
        with the largest absolute value.
        """
        u = []
        u.append( self.r - self.r )
        u.append( self.tp- self.t )
        u.append( self.rp- self.rp )

        antisymmetric = [np.linalg.norm(u0,ord=np.inf)<self.tol for u0 in u]
        return np.all( antisymmetric )

    def __mul__(self,other):
        new = DiagonalScatteringMatrix(self.n)
        new.tol = min([self.tol,other.tol])

        if self.n!=other.n:
            raise ValueError("Number of modes is incompatible")
        if not (self.unitary() and other.unitary()):
            raise ValueError("One of the matrices not unitary")

        rpr =-np.einsum('qij,qjk->qik',self.rp,other.r)
        rpr+= np.eye(self.sps)[None,:,:]
        rpr = np.einsum('qij,qjk->qik',np.linalg.inv(rpr),self.t)
        rrp = -np.einsum('qij,qjk->qik',other.r,self.rp)
        rrp+= np.eye(self.sps)[None,:,:]
        rrp = np.einsum('qij,qjk->qik',np.linalg.inv(rrp),other.tp)
        
        new.t = np.einsum('qij,qjk->qik',other.t,rpr)
        new.r = self.r + np.einsum('qij,qjk,qkl->qil',self.tp,other.r,rpr)
        new.tp= np.einsum('qij,qjk->qik',self.tp,rrp)
        new.rp= other.rp+ np.einsum('qij,qjk,qkl->qil',other.t,self.rp,rrp)
  
        if not new.unitary():
            raise ValueError("Result not unitary")

        return new

    def fano_factor(self):
        """
        Return the Fano factor given by
            F = Tr[ t*t (1-t*t) ]/Tr[t*t]
        where t* is the complex conjugate of t
        """
        ttH = np.einsum('qji,qjk->qik',np.conjugate(self.t),self.t)
        fano_factor = np.einsum('qij,qji',ttH,np.eye(self.sps)[None,:,:] - ttH)
        fano_factor*= 1/np.sum(np.einsum('qii',ttH))
        return np.real(fano_factor)

def matrix_exponent(R0):
    """
    For a orthongonal matrix R0 with det(R0)=+1, we can
    find a (non-unique) real skew-symmetric matrix A=-A.T
    such that R0=exp(A) -- this routine attempts to find
    A, although I do not think it always succeeds - hence the
    various ASSERT checks
    """
    assert np.linalg.norm(R0.T.dot(R0) - np.eye(R0.shape[0]))<1e-10
    assert np.isclose(la.det(R0),1)
    assert np.linalg.norm(np.imag(R0))<1e-10
    
    t,z = la.schur(np.real(R0))
    t0 = la.logm(t)
    h = np.real(t0)
    s = np.real(np.exp(1j*np.diag(np.imag(t0))))

    assert np.isclose(np.prod(s),1)

    ind0 = np.where(s<0)[0]
    for n in range(len(ind0)//2):
        h[ind0[2*n],ind0[2*n+1]] = np.pi
        h[ind0[2*n+1],ind0[2*n]] =-np.pi
    h = z.dot(h.dot(z.T))
    assert np.linalg.norm(R0 - la.expm(h))<1e-10
    return h
