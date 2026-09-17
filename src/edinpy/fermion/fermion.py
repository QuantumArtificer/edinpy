_dofs = None # global list of constructed degrees of freedom (DoF)
_N_vector = None # global list of attributes <dof>.N for nD -> 1D mapping
_fockspace = None # global var for constructed Fock space

def clear():
    ''' Clear the degrees of freedom and fock space stored as global variables. It is
        necessary to call this function when you want to change the model.
    '''
    global _dofs, _N_vector, _fockspace
    _dofs = None
    _N_vector = None
    _fockspace = None

_normal_chars = "0123456789"
_subscript_chars = "₀₁₂₃₄₅₆₇₈₉"
#_superscript_chars = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_mapping = str.maketrans(_normal_chars, _subscript_chars)

def _subscripts(text):
    ''' Converts subscript from normal text numbers to subscript numbers.
    '''
    converted_text = text.translate(_mapping)
    return converted_text

class DoF:
    ''' 
    Create a degree of freedom.
        
    Parameters
    ----------
    N_eigenvalues : int
        Number of eigenvalues of the associated observable, e.g. electron
        spin has 2 eigenvalues, a chain of 5 atoms has 5 position eigenvalues.
    name : string
        Name of the observable.
    '''

    def __init__(self, N_eigenvalues, name = None):
        self._N = N_eigenvalues # number of eigvals of observable
        self._name = name # given name of observable
        self._add_dof()

    def _add_dof(self):
        ''' 
        Add DoF to global vars lists.
        '''
        global _dofs, _N_vector

        # add attr N to a global list used to calculate effective sites
        if _N_vector is not None:
            _N_vector.append(_N_vector[-1] * _dofs[-1].N)
        else:
            _N_vector = [1]

        # add DoF to a global list
        if _dofs is not None:
            _dofs.append(self)
        else:
            _dofs = [self]

        self._label = len(_dofs) - 1 # numerical label of DoF

    @property
    def N(self):
        '''
        Number of eigenvalues set by parameter `N_eigenvalues`.
        :type: int
        '''
        return self._N
    
    @N.setter
    def N(self, N):
        self._N = N
    
    @property
    def name(self):
        ''' 
        Name given to the degree of freedom
        :type: string
        '''
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name
    
    @property
    def label(self):
        ''' 
        Numerical label given to the degree of freedom
        :type: int
        '''
        return self._label
    
    @label.setter
    def label(self, label):
        self._label = label

class model:
    ''' 
    Construct a model with the stored degrees of freedom.

    Parameters
    ----------
    Nf : int
        Number of fermions.
    auto : bool
        If auto is True then the Fock space is calculated automatically, otherwise
        the user must use the build() method to calculate it manually.
    '''

    def __init__(self, Nf, auto = True):
        global _dofs, _N_vector, _Nf

        self._dofs = _dofs
        self._Nvec = _N_vector
        self._Nf = _Nf = Nf

        if auto is True:
            self.build()
        
    def build(self):
        '''
        Build the Fock space with the given number of fermions and degrees of freedom.
        '''
        from numpy import prod
        from math import factorial as fact
        global _Neff, _Nbasis, _fockspace
        
        self._Neff = _Neff = prod([dof.N for dof in self._dofs])
        self._Nbasis = _Nbasis = int(fact(self._Neff)/(fact(self._Nf)*fact(self._Neff-self._Nf)))
        self._fockspace = _fockspace = fermionspace()
        _fockspace.build()
        self._fockspace.build()

    def activate(self):
        ''' 
        Set this model as the active instance.  
        '''
        global _dofs, _N_vector
        global _Neff, _Nf, _Nbasis, _fockspace

        _dofs = self._dofs
        _N_vector = self._Nvec
        _Neff = self._Neff
        _Nf = self._Nf
        _Nbasis = self._Nbasis
        _fockspace = self._fockspace

    @property
    def Neff(self):
        '''
        Number of effective sites.
        :type: int
        '''
        return self._Neff
    
    @Neff.setter
    def Neff(self, Neff):
        self._Neff = Neff

    @property
    def Nbasis(self):
        '''
        Number of basis states in the Fock space.
        :type: int
        '''
        return self._Nbasis
    
    @Nbasis.setter
    def Nbasis(self, Nbasis):
        self._Nbasis = Nbasis

    @property
    def fockspace(self):
        '''
        Class instance for the relevant Fock space of the current model.
        :type: fermionspace
        '''
        return self._fockspace
    
    @fockspace.setter
    def fockspace(self, fockspace):
        self._fockspace = fockspace

    @property
    def dofs(self):
        '''
        List of instances of degrees of freedom.
        :type: list
        '''
        return self._dofs
    
    @dofs.setter
    def dofs(self, dofs):
        self._dofs = dofs

  
class fockstate:
    ''' Create a fermionic Fock state.

        Parameters
        ----------
        state : int
            Occupation numbers of the effective sites represented as an integer.
        amp : scalar
            Probability amplitude of the state. Default value is 1.
        index : int
            Index of the state. Used when constructing a Fock space. Default is None.
    '''

    def __init__(self, state, amp = 1, index = None):
        from numpy import csingle
        self.state = state
        self.amp = csingle(amp)
        self.index = index
        self.bra = False

    def __pos__(self):
        return self

    def __neg__(self):
        return fockstate(self.state, amp = -self.amp, index = self.index)

    def __add__(self,fs):

        if isinstance(fs, fockstate):
            return statesum([self,fs])
        
        if isinstance(fs, null):
            return self
        
        if isinstance(fs, statesum):
            return statesum([self] + fs.states)
        
        raise NotImplementedError
    
    def __sub__(self,fs):

        if isinstance(fs, fockstate):
            return statesum([self,-fs])
        
        if isinstance(fs, null):
            return self
        
        if isinstance(fs, statesum):
            negstates = [-s for s in fs.states]
            return statesum([self] + negstates)
        
        raise NotImplementedError

    def __mul__(self,obj):
        
        if isinstance(obj, fockstate):
            if obj.state == self.state:
                if self.bra is True:
                    return self.amp * obj.amp
                else:
                    return self.amp.conj() * obj.amp
            else:
                return 0
            
        if isinstance(obj, null):
            return 0
        
        if isinstance(obj, scalar):
            if self.bra is False:
                amp = obj.value * self.amp.conj()
            else:
                amp = obj.value * self.amp
            newfs = fockstate( self.state, amp = amp)
            newfs.bra = True
            return newfs
        
        if isinstance(obj, _SingleModeOperator):
            product =  obj.dag*self
            product.bra = True
            if self.bra is False:
                product.amp = product.amp.conj()
            return product
        
        if isinstance(obj, operatorsum):
            sum = self*obj.os[0]
            for i in range(1,len(obj.os)):
                sum += self*obj.os[i]
            return sum
        
        if isinstance(obj, operatorproduct):
            product = self*obj.op[0]
            for i in range(1,len(obj.op)):
                product *= obj.op[i]
            return product
        
        if isinstance(obj,statesum):
            sum = 0
            for s in obj.states:
                sum += self*s
            return sum
        
        raise NotImplementedError

    def __eq__(self,fs):
        if fs.state == self.state:
            return True
        return False
        
    def __str__(self):
        return self.string
    
    def __format__(self, spec = None):
        return self.string 
    
    def is_occupied(self, _calc_effsite):
        '''
        Determine whether the effective site is occupied.
        :param _calc_effsite: effective site index
        :type _calc_effsite: int
        :rtype: bool
        '''
        return self.state & (1 << _calc_effsite)
    
    @property
    def string(self):
        ''' 
        Human readable representation of Fock state.
        :rtype: str
        '''

        global _Neff

        string = bin(self.state)[2:] # binary representation of s
        string = string[::-1] # order in terms of increasing site number from left to right
        n = len(string) # number of fermions
        # fill remaining sites as 0
        for i in range(_Neff-n):
            string+='0' 

        return f'{self.amp}|{string}>'
    
class vacuum(fockstate):
    ''' Create a vacuum state.

        Parameters
        ----------
        amp : complex
            Probability amplitude of the state. Default value is 1.
    '''

    def __init__(self, amp = 1):
        from numpy import csingle
        self.amp = csingle(amp)
        self.state = 0
        self.index = -1
        self.bra = False

    def __mul__(self, obj):
        
        if isinstance(obj, fockstate):

            if isinstance(obj, vacuum):
                if self.bra is True:
                    return self.amp * obj.amp
                else:
                    return self.amp.conj() * obj.amp
            
            return 0
        
        if isinstance(obj, null):
            return 0
        
        if isinstance(obj, scalar):
            if self.bra is False:
                self.amp = obj.value * self.amp.conj()
            else:
                self.amp = obj.value * self.amp
            self.bra = True
            return self
        
        if isinstance(obj, _SingleModeOperator):
            product =  obj.dag * self
            product.bra = True
            if self.bra is False:
                product.amp = product.amp.conj()
            return product
        
        if isinstance(obj, operatorsum):
            sum = self*obj.os[0]
            for i in range(1,len(obj.os)):
                sum += self*obj.os[i]
            return sum
        
        if isinstance(obj, operatorproduct):
            product = self*obj.op[0]
            for i in range(1,len(obj.op)):
                product = product * obj.op[i]
            return product
        
        if isinstance(obj,statesum):
            sum = 0
            for s in obj.states:
                sum += self*s
            return sum
        
        raise NotImplementedError

    def __eq__(self,ket):
        if isinstance(ket,vacuum):
            return True
        else:
            return False
    
    @property
    def string(self):
        ''' 
        Human readable representation of Fock state.
        :rtype: str
        '''
        return f'{self.amp}|{0}>'
    
class null():
    ''' 
    Additive identity for Fock states and operators.
    '''

    def __init__(self):
        from numpy import csingle
        self.amp = csingle(0)
        self.state = 0
        self.index = None

    def __add__(self,fs):
        return fs

    def __mul__(self,obj):

        if isinstance(obj, fockstate):
            return 0
        
        if isinstance(obj, scalar):
            return self
        
        if isinstance(obj, (_SingleModeOperator, operatorsum, operatorproduct)):
            return self
        
        if isinstance(obj, statesum):
            return 0
        
        raise NotImplementedError
    
    def __eq__(self,fs):
        if isinstance(fs,null):
            return True
        else:
            return False
        
    def __str__(self):
        return self.string
    
    def __format__(self, spec = None):
        return self.string
    
    @property
    def string(self):
        ''' 
        Human readable representation of Fock state.
        :rtype: str
        '''
        return '0'
    
class statesum:
    ''' 
    Sum of Fock states.

    Parameters
    ----------
    stateslist : list
        List containing Fock states.
    '''

    def __init__(self, stateslist):
        self.states = stateslist

    def __add__(self, obj):

        if isinstance(obj, fockstate):
            states = [s for s in self.states]
            states.append(obj)
            return statesum(states)
        
        if isinstance(obj, _SingleModeOperator):
            sum = self._os[0]*obj
            for i in range(1,len(self._os)):
                sum += self._os[i]*obj
            return sum
        
        if isinstance(obj, operatorsum):
            sum = self*obj.os[0]
            for i in range(1,len(obj.os)):
                sum += self*obj.os[i]
            return sum
        
        if isinstance(obj, operatorproduct):
            product = self*obj.op[0]
            for i in range(1,len(obj.op)):
                product *= obj.op[i]
            return product
        
        if isinstance(obj, statesum):
            return statesum(self.states + obj.states)
        
        if isinstance(obj, null):
            return self
        
        raise NotImplementedError
        
    def __mul__(self,obj):

        if isinstance(obj, fockstate):
            sum = 0
            for bra in self.states:
                sum += bra*obj
            return sum
        
        if isinstance(obj, scalar):
            sum = self._os[0] * obj
            for i in range(1, len(self._os[0])):
                sum += self._os[i] * obj
            return sum
        
        if isinstance(obj, statesum):
            sum = 0
            for bra in self.states:
                for ket in obj.states:
                    sum += bra*ket
            return sum
        
        raise NotImplementedError
        
    def __str__(self):
        return self.string

    def __format__(self,spec):
        return self.string
    
    @property
    def string(self):
        string = ''
        for i in range(len(self.states)-1):
            string += str(self.states[i]) + ' + '
        string += str(self.states[-1])
        return string
    
class fermionspace:
    ''' 
    Create a fermionic Fock space.
    '''

    def __init__(self):
        global _Nbasis, _Neff, _Nf

        self._Nbasis = _Nbasis
        self._Neff = _Neff
        self._Nf = _Nf
        self.ls = None

    def __contains__(self, s):
        from bisect import bisect_left
        
        i = bisect_left(self.ls, s.state)
        if i != len(self.ls) and self.ls[i] == s.state:
            return True, i
        else:
            return False
        
    contains = __contains__
    
    def build(self):
        '''
        Build the fock space basis vectors.
        '''
        self.basis  = [0 for i in range(self._Nbasis)]
        self.ls  = [0 for i in range(self._Nbasis)]
        self.basis[0] = fockstate( self.firstvector(), index = 0 ) # calculate the first state (state with lowest integer representation)
        self.ls[0] = self.basis[0].state
        # calculate the remaining states as lexicographical bit permutations
        for i in range(1,self._Nbasis):
            self.basis[i] = fockstate( self.nextvector( self.basis[i-1].state ), index = i)
            self.ls[i] = self.basis[i].state
        
    def firstvector(self):
        """ 
        Generate the first fermionic occupation vector in lexicographical order, i.e. the state with lowest integer representation.
        """
        v = ''
        # fill the last Neff-Nf effective sites with 0 fermions
        for i in range(self._Neff-self._Nf):
            v+= '0'
        # fill the first Nf effective sites with 1 fermion
        for i in range(self._Nf):
            v += '1'
        # return the integer representation of s
        return int(v,2)
    
    def nextvector(self,fv):
        """ 
        Calculate the next occupation vector in lexicographical order.
            
        Source:  https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation

        :param fv: a fermionic occupation vector
        :type fv: fockstate
        """
        t = (fv|(fv-1)) + 1
        return t|(((int(int(t & -t) / int(fv & -fv))) >> 1) - 1)
    
    def show(self):
        '''
        Print all the basis states.
        '''
        for s in self.basis:
            print(f'state {s.index}: {s}')
 
class scalar:
    ''' 
    Create a scalar.

    Parameters
    ----------
    value : scalar
        The numerical value of the scalar
    label : str
        A string label for the scalar
    '''

    def __init__(self, value, label):
        self.value = value
        self.label = label

    def __neg__(self):
            return scalar(-1, f'({-1})') * self
    
    def __pos__(self):
        return self

    def __add__(self, obj):

        if isinstance(obj, scalar):
            return operatorsum([self, obj])
        
        if isinstance(obj, operatorsum):
            os = [self]
            os += obj.os
            return operatorsum(os)

        if isinstance(obj, operatorproduct):
            return operatorsum([self, obj])
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorsum([self, sclr])

        raise NotImplementedError
    
    def __radd__(self, obj):
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorsum([sclr, self])
        
        raise NotImplementedError
    
    def __sub__(self, obj):
        negone = scalar(-1,f'({-1})')
        
        if isinstance(obj, scalar):
            negobj = negone * obj
            return operatorsum([self, negobj])
        
        if isinstance(obj, (int, float, complex)):
            
            sclr = scalar(obj, f'{obj}')
            sclr = negone * sclr
            return operatorsum([self, sclr])

        raise NotImplementedError
    
    def __rsub__(self, obj):
        negone = scalar(-1,f'({-1})')
        
        if isinstance(obj, (int, float, complex)):
            
            sclr = scalar(obj, f'{obj}')
            negself = negone * self
            return operatorsum([sclr, negself])

        raise NotImplementedError

    def __mul__(self, obj):
        
        if isinstance(obj, fockstate):
            
            if isinstance(obj, vacuum):
                obj.amp *= self.value
                return obj
            else:
                amp = self.value * obj.amp
                return fockstate(obj.state, amp = amp)
            
        if isinstance(obj, null):
            return obj
        
        if isinstance(obj, _SingleModeOperator):
            prod = [self, obj]
            return operatorproduct(prod)

        if isinstance(obj, operatorproduct):
            prod = [self] + obj.op
            return operatorproduct(prod)  

        if isinstance(obj, operatorsum):
            sum = self * obj.os[0]
            for i in range(1,len(obj.os)):
                sum += self * obj.os[i]
            return sum
        
        if isinstance(obj, statesum):
            sum = self * obj.states[0]
            for i in range(1,len(obj.states)):
                sum += self * obj.states[i]
            return sum
        
        if isinstance(obj, scalar):
            value = self.value * obj.value
            label = self.label + obj.label
            return scalar(value, label)
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')

            return operatorproduct([sclr, self])
        
        raise NotImplementedError
    
    def __rmul__(self, obj):
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorproduct([sclr, self])
        
        raise NotImplementedError
    
    def __str__(self):
        return f'{self.value}'
    
    def __format__(self, spec):
        return f'{self.label}'
    
    @property
    def string(self):
        '''
        Return the label of the scalar.
        '''
        return f'{self.label}'
    
class Operator:
    '''Base class for symbolic fermionic operators.'''
    pass

class _SingleModeOperator(Operator):
    ''' 
    Create an annihilation operator.

    Parameters
    ----------
    indices : iterable or int
        The indices which the operator acts on. If `indices` is an `int` then it is assumed 
        to refer to the effective site.
    '''

    def __init__(self, indices):
        self._indices = indices
        self._calc_effsite()

    def __pos__(self):
        return self

    def __neg__(self, obj):
        negone = scalar(-1, f'(-1)')
        return negone * self

    def __add__(self, obj):

        if isinstance(obj, (_SingleModeOperator, operatorproduct)):
            return operatorsum([self, obj])
        
        if isinstance(obj, operatorsum):
            os = [0 for i in range(len(obj.os)+1)]
            os[0] = self
            for i, o in enumerate(obj.os):
                os[i+1] = o
            return operatorsum(os)
        
        if isinstance(obj, scalar):
            return operatorsum([self, obj])
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorsum([self, sclr])
        
        raise NotImplementedError
    
    def __radd__(self, obj):

        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorsum([sclr, self])
        
        raise NotImplementedError
    
    def __sub__(self, obj):
        
        negone = scalar(-1,f'({-1})')
        
        if isinstance(obj, (_SingleModeOperator, operatorproduct)):
            return operatorsum([self, negone * obj])
        
        if isinstance(obj, operatorsum):
            os = [0 for i in range(len(obj.os)+1)]
            os[0] = self
            for i, o in enumerate(obj.os):
                os[i+1] = negone * o
            return operatorsum(os)
        
        if isinstance(obj, scalar):
            return operatorsum([self, negone * obj])
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorsum([self, negone * sclr])
        
        raise NotImplementedError
    
    def __rsub__(self, obj):
        negone = scalar(-1,f'({-1})')
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorsum([obj, negone * self])
        
        raise NotImplementedError
    
    def __mul__(self,obj):
        
        if isinstance(obj,fockstate):
            return self._action(obj)
        
        if isinstance(obj,null):
            return obj
        
        if isinstance(obj, scalar):
            return operatorproduct([obj, self])
        
        if isinstance(obj,_SingleModeOperator):
            return operatorproduct([self,obj])

        if isinstance(obj,operatorsum):
            os = [0 for i in range(len(obj.os))]
            for i, o in enumerate(obj.os):
                os[i] = self*o
            return operatorsum(os)

        if isinstance(obj,operatorproduct):
            op = [ o for o in obj.op ]
            op = [self] + op
            return operatorproduct(op)

        if isinstance(obj,statesum):
            sum = self*obj.states[0]
            for i in range(1,len(obj.states)):
                sum += self*obj.states[i]
            return sum
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorproduct([sclr,self])
        
        raise NotImplementedError
    
    def __rmul__(self, obj):
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorproduct([sclr,self])
        
        raise NotImplementedError

    def __eq__(self, obj):
        raise NotImplementedError
        
    def __str__(self):
        return self.string
    
    def __format__(self,spec):     
        return self.string

    def _annihilate(self,fs):
        """ 
        Annihilate a particle in a Fock state.

        :param fs: a fermionic Fock state
        :type fs: fockstate
        """
        
        if isinstance(fs,vacuum):
            return null()
        
        # Annihilate a fermion in the i-th mode.
        if fs.is_occupied(self._site):
            bit2flip = 1 << self._site

            # Fermionic sign from occupied modes preceding the acted-on mode.
            preceding_mask = bit2flip - 1
            parity = (fs.state & preceding_mask).bit_count() & 1
            sign = -1 if parity else 1

            newstate = fs.state ^ bit2flip
            amp = sign * fs.amp

            if newstate == 0:
                return vacuum(amp=amp)
            return fockstate(newstate, amp=amp)

        return null()
        
    def _action(self,fs):
        '''
        _action of the annihilation operator on a Fock state.

        :param fs: a fermionic Fock state
        :type fs: fockstate
        '''
        return self._annihilate(fs)
        
    def _checkdims(self):
        if len(self._indices) != len(_dofs):
            raise Exception(f'The number of indices {len(self._indices)} exceeds the number of degrees of freedom {len(_dofs)}.')
        
        for i,d in enumerate(_dofs):
            if self._indices[i] >= d.N:
                nameorlabel = d.name if (d.name != None) else d.label
                raise Exception(f'The index {self._indices[i]} at position {i} is out of bounds for DoF [{nameorlabel}] of size {d.N-1} (0 < index < {d.N}).')
            
    def _calc_effsite(self):
        '''
        Calculate the effective site corresponding the given indices.
        '''
        if isinstance(self._indices,int):
            if self._indices >= _Neff:
                raise Exception(f'The index {self._indices} is out of bounds for the number of effective sites {_Neff} (0 < index < {_Neff})')
            self._site = self._indices
            return
        
        from numpy import array
        self._checkdims()
        self._site = int(array(_N_vector).dot(self._indices))
        
    @property
    def dag(self):
        '''
        Return the Hermitian adjoint of the operator.
        '''
        return dagger(self._indices, self._site)
    
    @property
    def indices(self):
        '''
        The indices which the operator acts on.
        :rtype: iterable or int
        '''
        return self._indices
    
    @indices.setter
    def indices(self, indices):
        self._indices = indices
    
    @property
    def site(self):
        '''
        Effective site that the operator acts on.
        :rtype: int
        '''
        return self._site
    
    @site.setter
    def site(self, effsite):
        self._site = effsite
    
    @property
    def string(self):
        '''
        Return the string representing the operator.
        '''
        if isinstance(self._indices, int):
            string = str(self._indices)
        else:
            string = ''
            for i in range(len(self._indices)-1):
                string += str(self._indices[i]) + ','
            string += str(self._indices[-1])    
        return 'c' + _subscripts(string)
    
class Annihilation(_SingleModeOperator):
    '''Fermionic annihilation operator.'''
    pass

# Backward-compatible historical API.
operator = Annihilation

class Creation(_SingleModeOperator):
    ''' 
    Hermitian adjoint of an operator.

    Parameters
    ----------
    indices : iterable or int
        The indices which the operator acts on. If `indices` is an `int` then it is assumed 
        to refer to the effective site.
    site : int
        Effective site that the operator acts on.
    '''

    def __init__(self, indices, site=None):
        self._indices = indices
        if site is None:
            self._calc_effsite()
        else:
            self._site = site
    
    def __format__(self,spec):
        return self.string

    def __eq__(self, obj):
        if isinstance(obj,dagger):
            if obj.site == self._site:
                return True
        return False
    
    def _create(self,fs):
        """ Creation operator acting on the i-th site. Notation: state s -> (s,p) where p is probability amplitude
        """
        
        if isinstance(fs, vacuum):
            bit2flip = 1 << self._site
            return fockstate(bit2flip, amp=fs.amp)

        # Create a fermion in the i-th mode.
        if not fs.is_occupied(self._site):
            bit2flip = 1 << self._site

            # Fermionic sign from occupied modes preceding the acted-on mode.
            preceding_mask = bit2flip - 1
            parity = (fs.state & preceding_mask).bit_count() & 1
            sign = -1 if parity else 1

            return fockstate(
                fs.state ^ bit2flip,
                amp=sign * fs.amp,
            )

        return null()
        
    def _action(self,obj):
        return self._create(obj)
        
    @property
    def dag(self):
        '''
        Return the Hermitian adjoint of the operator.
        '''
        return operator(self._indices)
    
    @property
    def string(self):
        '''
        Return the string representing the operator.
        '''
        if isinstance(self._indices, int):
            string = str(self._indices)
        else:
            string = ''
            for i in range(len(self._indices)-1):
                string += str(self._indices[i]) + ','
            string += str(self._indices[-1])    
        return 'c†' + _subscripts(string)
    
# Backward-compatible historical API.
dagger = Creation

class OperatorSum(Operator):
    ''' 
    Sum of operators.

    Parameters
    ----------
    operatorlist : list
        A list containing the operators that are summed.
    '''

    def __init__(self, operatorlist):
        self._os = operatorlist

    def __pos__(self):
        return self

    def __neg__(self):
        os = [-o for o in self._os]
        return operatorsum(os)

    def __add__(self, obj):

        if isinstance(obj, (scalar, _SingleModeOperator, operatorproduct)):
            os = [o for o in self._os]
            os.append(obj)
            return operatorsum(os)
        
        if isinstance(obj, operatorsum):
            return operatorsum(self._os + obj.os)
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return self + sclr
        
        raise NotImplementedError
    
    def __radd__(self, obj):
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return sclr + self
        
        raise NotImplementedError
    
    def __sub__(self, obj):
        negone = scalar(-1,f'({-1})')

        if isinstance(obj, (scalar, _SingleModeOperator, operatorproduct)):
            os = [o for o in self._os]
            os.append(negone * obj)
            return operatorsum(os)
        
        if isinstance(obj, operatorsum):
            negobj = negone * obj
            return operatorsum(self._os + negobj.os)
        
        if isinstance(obj, (int, float, complex)):
            sclr = negone * scalar(obj, f'{obj}')
            return self + sclr
        
        raise NotImplementedError
    
    def __rsub__(self, obj):
        negone = scalar(-1,f'({-1})')
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return sclr + negone*self
        
        raise NotImplementedError
        
    def __mul__(self, obj):
        
        if isinstance(obj, fockstate):
            sum = self._os[0]*obj
            for i in range(1,len(self._os)):
                sum += self._os[i]*obj
            return sum
        
        if isinstance(obj, scalar):
            sum = self._os[0] * obj
            for i in range(1, len(self._os)):
                sum += self._os[i] * obj
            return sum
        
        if isinstance(obj, (_SingleModeOperator,operatorproduct)):
            os = [0 for i in range(len(self._os))]
            for i, o in enumerate(self._os):
                os[i] = o*obj
            return operatorsum(os)
        
        if isinstance(obj, operatorsum):
            sum = self._os[0]*obj
            for i in range(1,len(self._os)):
                sum += self._os[i]*obj
            return sum
        
        if isinstance(obj, statesum):
            sum = self._os[0]*obj
            for i in range(1,len(self._os)):
                sum += self._os[i]*obj
            return sum
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return self * sclr

        raise NotImplementedError
    
    def __rmul__(self, obj):
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return sclr * self

        raise NotImplementedError
        
    def __eq__(self,obj):
        raise NotImplementedError
        
    def __str__(self):
        return self.string
    
    def __format__(self, spec):
        return self.string
    
    @property
    def dag(self):
        '''
        Return the Hermitian adjoint of the sum of operators.
        '''
        listdag = [0 for o in self._os]
        for i, o in enumerate(self._os):
            listdag[i] = o.dag
        return operatorsum(listdag)
    
    @property
    def os(self):
        '''
        Return the list of summed operators
        '''
        return self._os

    @os.setter
    def os(self, operatorlist):
        self._os = operatorlist

    @property
    def string(self):
        '''
        Return the string representing the sum of operators.
        '''
        string = ''
        for i in range(len(self._os)-1):
            string += self._os[i].string + ' + '
        string += self._os[-1].string
        return string
        
class OperatorProduct(Operator):
    ''' 
    Product of operators.

    Parameters
    ----------
    operatorlist : list
        A list containing the operators that are summed.
    '''

    def __init__(self, operatorlist):
        self._op = operatorlist

    def __pos__(self):
        return self

    def __neg__(self):
        negone = scalar(-1, '(-1)')
        return operatorproduct([negone] + self._op)

    def __add__(self, obj):

        if isinstance(obj, (scalar, _SingleModeOperator, operatorproduct)):
            return operatorsum([self, obj])
        
        if isinstance(obj, operatorsum):
            os = [0 for i in range(len(obj.os)+1)]
            os[0] = self
            for i, o in enumerate(obj.os):
                os[i+1] = o
            return operatorsum(os)
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorsum([self, sclr])
        
        raise NotImplementedError
    
    def __radd__(self, obj):
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorsum([sclr, self])
        
        raise NotImplementedError
    
    def __sub__(self, obj):
        negone = scalar(-1,f'({-1})')

        if isinstance(obj, (scalar, _SingleModeOperator, operatorproduct)):
            return operatorsum([self, negone * obj])
        
        if isinstance(obj, operatorsum):
            negobj = negone * obj
            os = [self]
            os += negobj.os
            return operatorsum(os)
        
        if isinstance(obj, (int, float, complex)):
            sclr = negone * scalar(obj, f'{obj}')
            return operatorsum([self, sclr])
        
        raise NotImplementedError
    
    def __rsub__(self, obj):
        negone = scalar(-1,f'({-1})')
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            negself = negone * self
            return operatorsum([sclr, negself])
        
        raise NotImplementedError

    def __mul__(self,obj):

        if isinstance(obj, fockstate):
            product = self._op[-1]*obj
            for i in range(len(self._op)-2,-1,-1):
                product = self._op[i]*product
            return product
        
        if isinstance(obj, scalar):
            return operatorproduct([obj] + self._op)

        if isinstance(obj, _SingleModeOperator):
            op = [o for o in self._op]
            op.append(obj)
            return operatorproduct(op)

        if isinstance(obj, operatorproduct):
            return operatorproduct(self._op + obj.op)

        if isinstance(obj, operatorsum):
            os = [0 for i in range(len(obj.os))]
            for i, o in enumerate(obj.os):
                os[i] = self*o
            return operatorsum(os)

        if isinstance(obj, statesum):
            sum = self*obj.states[0]
            for i in range(1,len(obj.states)):
                sum += self*obj.states[i]
            return sum
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorproduct([sclr] + self._op)
        
        raise NotImplementedError
    
    def __rmul__(self, obj):

        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorproduct([sclr] + self._op)
        
        raise NotImplementedError
        
    def __eq__(self, obj):
        raise NotImplementedError
    
    def __str__(self):
        return self.string
    
    def __format__(self, spec):
        return self.string
    
    @property
    def dag(self):
        listdag = [0 for o in self._op]
        for i, o in enumerate(self._op):
            listdag[i] = o.dag
        listdag = list(reversed(listdag))
        return operatorproduct(listdag)
    
    @property
    def string(self):
        '''
        Return the string representing the product of operators.
        '''
        string = ''
        for i in range(len(self._op)):
            string += self._op[i].string
        return string
    
    @property
    def op(self):
        '''
        Return the list of multiplied operators
        '''
        return self._op

    @op.setter
    def op(self, operatorlist):
        self._op = operatorlist
    
# Backward-compatible historical API.
operatorsum = OperatorSum
operatorproduct = OperatorProduct

class Number(_SingleModeOperator):
    ''' 
    Create a number operator.

    Parameters
    ----------
    indices : iterable or int
        The indices which the operator acts on. If `indices` is an `int` then it is assumed 
        to refer to the effective site.
    '''

    def __init__(self, indices):
        self._indices = indices
        self._calc_effsite()

    def __pos__(self):
        return self

    def __neg__(self):
        negone = scalar(-1, '(-1)')
        return operatorproduct([negone,self])

    def _count(self,fs):
        """ 
        Annihilate a particle in a Fock state.

        :param fs: a fermionic Fock state
        :type fs: fockstate
        """
        
        if isinstance(fs,vacuum):
            return null()
        
        # Annihilate fermion in i-th site
        if fs.state & (1<<self._site): # if the site occupation number n_i = 1
            return fockstate(fs.state, amp = fs.amp)
        else:
            # otherwise annihilate the state
            return null()
        
    def _action(self,fs):
        return self._count(fs)
    
    def __str__(self):
        return self.string
    
    def __format__(self,spec):    
        return self.string
    
    @property
    def dag(self):
        '''
        Return the Hermitian adjoint of the number operator.
        '''
        return self
    
    @property
    def string(self):
        '''
        Return the string representing the number operator.
        '''
        if isinstance(self._indices, int):
            string = str(self._indices)
        else:
            string = ''
            for i in range(len(self._indices)-1):
                string += str(self._indices[i]) + ','
            string += str(self._indices[-1])    
        return 'n' + _subscripts(string)
    
# Backward-compatible historical API.
number = Number

class hamiltonian:
    ''' Create a hamiltonian.
        
        Parameters
        ----------
        obj : operator, operatorsum, operatorproduct or hamiltonian
            Operator form of the hamiltonian
    '''
    
    def __init__(self, obj):

        if isinstance(obj, hamiltonian):
            self.opform = obj.opform
            self.dense_matrix = obj.dense_matrix
            self.sparse_matrix = obj.sparse_matrix
            self._eigvals = obj.eigvals
            self._eigvecs = obj.eigvecs
        else:
            self.opform = obj
            self.dense_matrix = None
            self.sparse_matrix = None
            self._eigvals = None
            self._eigvecs = None

    def __add__(self, obj):

        raise NotImplementedError
    
    def __mul__(self, obj):

        raise NotImplementedError

    def calc_matrix(self):
        '''
        Calculate the sparse matrix representation of the Hamiltonian.

        Common symbolic operator terms are lowered to direct execution
        kernels. Unrecognized terms retain the general Fock-algebra path.
        '''
        from bisect import bisect_left
        from numpy import csingle
        from scipy.sparse import coo_array
        from ._execution import compile_operator
        global _Nbasis, _fockspace

        rows = []
        columns = []
        data = []

        basis_states = _fockspace.ls
        compiled = compile_operator(self.opform)

        for column, ket in enumerate(_fockspace.basis):
            output = compiled.apply(ket)

            for state, amplitude in output.items():
                row = bisect_left(basis_states, state)

                # Ignore states outside the active fixed-particle sector.
                if row == _Nbasis or basis_states[row] != state:
                    continue

                if amplitude != 0:
                    rows.append(row)
                    columns.append(column)
                    data.append(amplitude)

        self.sparse_matrix = coo_array(
            (data, (rows, columns)),
            shape=(_Nbasis, _Nbasis),
            dtype=csingle,
        ).tocsr()

        self.sparse_matrix.sum_duplicates()

    def eigsolve(self, sparse = True, numeigs = 2, which = 'SA'):
        '''
        Calculate the eigenvalues and eigenvectors of the Hamiltonian.

        Parameters
        ----------
        sparse : bool
            Determine whether to calculate the eigensystem using sparse or dense linear algebra.
        numeigs : int
            Number of eigenvalues and eigenvectors to return. Set to `'all'` to return all eigenvalues. 
        which : str
            Determine the order in which to calculate the eigenvalues. See `scipy.sparse.linalg.eigsh`.
        '''
        if numeigs == 'all':
            numeigs = _Nbasis

        if sparse is True:
            from scipy.sparse.linalg import eigsh
            self._eigvals, self._eigvecs = eigsh(self.sparse, k = numeigs, which = which)
        else:        
            from numpy.linalg import eigh
            self._eigvals, self._eigvecs = eigh(self.array)
            self._eigvals = self._eigvals[:numeigs]
            self._eigvecs = self._eigvecs[:, :numeigs]

    @property
    def array(self):
        '''
        Return the dense matrix form of the Hamiltonian. Calculates the matrix if
        it hasn't been already.
        :rtype: np.array
        '''
        if self.dense_matrix is None:
            if self.sparse_matrix is None:
                self.calc_matrix()
            self.dense_matrix = self.sparse_matrix.toarray()
        return self.dense_matrix
    
    @property
    def sparse(self):
        '''
        Return the sparse matrix form of the Hamiltonian. Calculates the matrix if
        it hasn't been already.
        :rtype: scipy.sparse.csr_matrix
        '''
        if self.sparse_matrix is None:
            self.calc_matrix()
        return self.sparse_matrix
    
    @property
    def eigvals(self):
        '''
        Return the calculated eigvals of the Hamiltonian.
        '''
        if self._eigvals is None:
            raise AttributeError('The eigensolutions must be calculated first')
        return self._eigvals
    
    @property
    def eigvecs(self):
        '''
        Return the calculated eigvecs of the Hamiltonian.
        '''
        if self._eigvecs is None:
            raise AttributeError('The eigensolutions must be calculated first')
        return self._eigvecs

        
        
    
