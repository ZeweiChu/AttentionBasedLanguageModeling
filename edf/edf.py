import numpy as np

DT=np.float32
eps=1e-12
# Globals
components = []
params = []

# Global forward/backward
def Forward():
    for c in components: c.forward()

def Backward(loss):
    for c in components:
        if c.grad is not None: c.grad = DT(0)
    loss.grad = np.ones_like(loss.value)
    for c in components[::-1]: c.backward();

# Optimization functions
def SGD(lr):
    for p in params:
        lrp = p.opts['lr']*lr if 'lr' in p.opts.keys() else lr
        p.value = p.value - lrp*p.grad
        p.grad = DT(0)

# Values
class Value:
    def __init__(self,value=None):
        self.value = DT(value).copy()
        self.grad = None

    def set(self,value):
        self.value = DT(value).copy()

# Parameters
class Param:
    def __init__(self,value,opts = {}):
        self.value = DT(value).copy()
        self.opts = {}
        params.append(self)
        self.grad = DT(0)

# Xavier initializer
def xavier(shape):
    sq = np.sqrt(3.0/np.prod(shape[:-1]))
    return np.random.uniform(-sq,sq,shape)


# Utility function for shape inference with broadcasting
def bcast(x,y):
    xs = np.array(x.shape)
    ys = np.array(y.shape)
    pad = len(xs)-len(ys)
    if pad > 0:
        ys = np.pad(ys,[[pad,0]],'constant')
    elif pad < 0:
        xs = np.pad(xs,[[-pad,0]],'constant')
    os = np.maximum(xs,ys)
    xred = tuple([idx for idx in np.where(xs < os)][0])
    yred = tuple([idx for idx in np.where(ys < os)][0])
    return xred,yred

'''
  function name: _im2c
  function usage: Reshape the tensor value to specific shape fidx and pick the valid pixel.
'''
def _im2c(value,fidx,vld):
    if vld is not None:
        fmat = np.zeros(np.prod(fidx.shape),dtype=DT)
        fmat[vld] = value.reshape([-1])[fidx.reshape([-1])[vld]]
    else:
        fmat = value.reshape([-1])[fidx.reshape([-1])]
    fmat = fmat.reshape(fidx.shape)
    return fmat


################################################### Actual components #####################################################

'''
  Class name: Add
  Class usage: add two matrices x, y with broadcasting supported by numpy "+" operation.
  Class function:
      forward: calculate x + y with possible broadcasting
      backward: calculate derivative w.r.t to x and y, when calculate the derivative w.r.t to y, we sum up all the axis over grad except the last dimension.
'''
class Add: # Add with broadcasting
    def __init__(self,x,y):
        components.append(self)
        self.x = x
        self.y = y
        self.grad = None if x.grad is None and y.grad is None else DT(0)

    def forward(self):
        self.value = self.x.value + self.y.value

    def backward(self):
        xred,yred = bcast(self.x.value,self.y.value)
        if self.x.grad is not None:
            self.x.grad = self.x.grad + np.reshape(
                np.sum(self.grad,axis=xred,keepdims=True),
                self.x.value.shape)

        if self.y.grad is not None:
            self.y.grad = self.y.grad + np.reshape(
                np.sum(self.grad,axis=yred,keepdims=True),
                self.y.value.shape)

'''
Class Name: Mul
Class Usage: elementwise multiplication with two matrix 
Class Functions:
    forward: compute the result x*y
    backward: compute the derivative w.r.t x and y
'''            
class Mul: # Multiply with broadcasting
    def __init__(self,x,y):
        components.append(self)
        self.x = x
        self.y = y
        self.grad = None if x.grad is None and y.grad is None else DT(0)

    def forward(self):
        self.value = self.x.value * self.y.value

    def backward(self):
        xred,yred = bcast(self.x.value,self.y.value)
        if self.x.grad is not None:
            self.x.grad = self.x.grad + np.reshape(
                np.sum(self.grad*self.y.value,axis=xred,keepdims=True),
                self.x.value.shape)

        if self.y.grad is not None:
            self.y.grad = self.y.grad + np.reshape(
                np.sum(self.grad*self.x.value,axis=yred,keepdims=True),
                self.y.value.shape)


'''
Class Name: VDot
Class Usage: matrix multiplication where x, y are matrices
y is expected to be a parameter and there is a convention that parameters come last. Typical usage is x is batch feature vector with shape (batch_size, f_dim), y a parameter with shape (f_dim, f_dim2).
Class Functions:
     forward: compute the vector matrix multplication result
     backward: compute the derivative w.r.t x and y, where derivative of x and y are both matrices 
'''            
class VDot: # Matrix multiply (fully-connected layer)
    def __init__(self,x,y):
        components.append(self)
        self.x = x
        self.y = y
        self.grad = None if x.grad is None and y.grad is None else DT(0)

    def forward(self):
        self.value = np.matmul(self.x.value,self.y.value)
    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + np.matmul(self.y.value,self.grad.T).T
        if self.y.grad is not None:
            self.y.grad = self.y.grad + np.matmul(self.x.value.T,self.grad)


'''
Class Name: Sigmoid
Class Usage: compute the elementwise sigmoid activation. Input is vector or matrix. In case of vector, [x_{0}, x_{1}, ..., x_{n}], output is vector [y_{0}, y_{1}, ..., y_{n}] where y_{i} = 1/(1 + exp(-x_{i}))
Class Functions:
    forward: compute activation y_{i} for all i.
    backward: compute the derivative w.r.t input vector/matrix x  
'''            
class Sigmoid:
    def __init__(self,x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = 1. / (1. + np.exp(-self.x.value))

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad * self.value * (1.-self.value)

'''
Class Name: Tanh
Class Usage: compute the elementwise Tanh activation. Input is vector or matrix. In case of vector, [x_{0}, x_{1}, ..., x_{n}], output is vector [y_{0}, y_{1}, ..., y_{n}] where y_{i} = (exp(x_{i}) - exp(-x_{i}))/(exp(x_{i}) + exp(-x_{i}))
Class Functions:
    forward: compute activation y_{i} for all i.
    backward: compute the derivative w.r.t input vector/matrix x  
'''            
class Tanh:
    def __init__(self,x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):

        x_exp = np.exp(self.x.value)
        x_neg_exp = np.exp(-self.x.value)

        self.value = (x_exp - x_neg_exp)/(x_exp + x_neg_exp)

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad * (1 - self.value*self.value)

'''
Class Name: RELU
Class Usage: compute the elementwise RELU activation. Input is vector or matrix. In case of vector, [x_{0}, x_{1}, ..., x_{n}], output is vector [y_{0}, y_{1}, ..., y_{n}] where y_{i} = max(0, x_{i})
Class Functions:
    forward: compute activation y_{i} for all i.
    backward: compute the derivative w.r.t input vector/matrix x  
'''            
class RELU:
    def __init__(self,x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = np.maximum(self.x.value,0)

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad * (self.value > 0)


'''
Class Name: LeakyRELU
Class Usage: compute the elementwise LeakyRELU activation. Input is vector or matrix. In case of vector, [x_{0}, x_{1}, ..., x_{n}], output is vector [y_{0}, y_{1}, ..., y_{n}] where y_{i} = 0.01*x_{i} for x_{i} < 0 and y_{i} = x_{i} for x_{i} > 0
Class Functions:
    forward: compute activation y_{i} for all i.
    backward: compute the derivative w.r.t input vector/matrix x  
'''            
class LeakyRELU:
    def __init__(self,x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):

        self.value = np.maximum(self.x.value, 0.01*self.x.value)

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad * np.maximum(0.01, self.value > 0)

'''
Class Name: Softplus
Class Usage: compute the elementwise Softplus activation.
Class Functions:
    forward: compute activation y_{i} for all i.
    backward: compute the derivative w.r.t input vector/matrix x  
'''             
class Softplus:
    def __init__(self,x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = np.log(1. + np.exp(self.x.value))

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad * 1./(1. + np.exp(-self.x.value))

'''
Class Name: SoftMax
Class Usage: compute the softmax activation for each element in the matrix, normalization by each all elements in each batch (row). Specificaly, input is matrix [x_{00}, x_{01}, ..., x_{0n}, ..., x_{b0}, x_{b1}, ..., x_{bn}], output is a matrix [p_{00}, p_{01}, ..., p_{0n},...,p_{b0},,,p_{bn} ] where p_{bi} = exp(x_{bi})/(exp(x_{b0}) + ... + exp(x_{bn}))
Class Functions:
    forward: compute probability p_{bi} for all b, i.
    backward: compute the derivative w.r.t input matrix x 
'''            
class SoftMax:
    def __init__(self,x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        lmax = np.max(self.x.value,axis=-1,keepdims=True)
        ex = np.exp(self.x.value - lmax)
        self.value = ex / np.sum(ex,axis=-1,keepdims=True)

    def backward(self):
        if self.x.grad is None:
            return
        gvdot = np.matmul(self.grad[...,np.newaxis,:],self.value[...,np.newaxis]).squeeze(-1)
        self.x.grad = self.x.grad + self.value * (self.grad - gvdot)

        
'''
Class Name: LogLoss
Class Usage: compute the elementwise -log(x) given matrix x. this is the loss function we use in most case.
Class Functions:
    forward: compute -log(x)
    backward: compute the derivative w.r.t input matrix x
'''        
class LogLoss:
    def __init__(self,x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = -np.log(np.maximum(eps,self.x.value))

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + (-1)*self.grad/np.maximum(eps,self.x.value)

'''
Class Name: Mean
Class Usage: compute the mean given a vector x.
Class Functions:
    forward: compute (x_{0} + ... + x_{n})/n
    backward: compute the derivative w.r.t input vector x
'''           
class Mean:
    def __init__(self,x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = np.mean(self.x.value)

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad*np.ones_like(self.x.value)/self.x.value.shape[0]

'''
Class Name: MeanwithMask
Class Usage: compute the mean given a vector x with mask.
Class Functions:
    forward: compute x = x*mask and then sum over nonzeros in x/#(nozeros in x)
    backward: compute the derivative w.r.t input vector matrix
'''             
class MeanwithMask:
    def __init__(self,x, mask):
        components.append(self)
        self.x = x
        self.mask = mask
        self.grad = None if x.grad is None else DT(0)

    def forward(self): 
        self.value = np.sum(self.x.value*self.mask.value)/np.sum(self.mask.value)
    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad*np.ones_like(self.x.value)*self.mask.value/np.sum(self.mask.value)

'''
Class Name: Aref
Class Usage: get some specific entry in a matrix. x is the matrix with shape (batch_size, N) and idx is vector contains the entry index and x is differentiable.
Class Functions:
    forward: compute x[b, idx(b)]
    backward: compute the derivative w.r.t input matrix x
'''            
class Aref: # out = x[idx]
    def __init__(self,x,idx):
        components.append(self)
        self.x = x
        self.idx = idx
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        xflat = self.x.value.reshape(-1)
        iflat = self.idx.value.reshape(-1)
        outer_dim = len(iflat)
        inner_dim = len(xflat)/outer_dim
        self.pick = np.int32(np.array(range(outer_dim))*inner_dim+iflat)
        self.value = xflat[self.pick].reshape(self.idx.value.shape)

    def backward(self):
        if self.x.grad is not None:
            grad = np.zeros_like(self.x.value)
            gflat = grad.reshape(-1)
            gflat[self.pick] = self.grad.reshape(-1)
            self.x.grad = self.x.grad + grad

'''
Class Name: Accuracy
Class Usage: check the predicted label is correct or not. x is the probability vector where each probability is for each class. idx is ground truth label.
Class Functions:
    forward: find the label that has maximum probability and compare it with the ground truth label.
    backward: None 
'''            
class Accuracy:
    def __init__(self,x,idx):
        components.append(self)
        self.x = x
        self.idx = idx
        self.grad = None

    def forward(self):
        self.value = np.mean(np.argmax(self.x.value,axis=-1)==self.idx.value)

    def backward(self):
        pass

'''
  Class name: Reshape
  Class usage: Reshape the tensor x to specific shape.
  Class function:
      forward: Reshape the tensor x to specific shape
      backward: calculate derivative w.r.t to x, which is simply reshape the income gradient to x's original shape
'''
class Reshape:
    def __init__(self,x,shape):
        components.append(self)
        self.x = x
        self.shape = shape
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = np.reshape(self.x.value,self.shape)
    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + np.reshape(self.grad,self.x.value.shape)
            
'''
  Class name: Conv
  Class usage: convolution layer given image feature f and filter k, stride and pad. this is use the image to column trick and fast. 
  Class function:
      forward: do concolution
      backward: calculate derivative w.r.t to f and filter k
      
'''
class Conv:
    def __init__(self,f,k,stride=1,pad=0):
        components.append(self)
        self.f = f
        self.k = k
        pad = np.array(pad)
        if pad.shape == ():
            self.xpad = self.ypad = pad
        else:
            self.ypad = pad[0]
            self.xpad = pad[1]
        self.stride=stride
        self.grad = None if f.grad is None and k.grad is None else DT(0)

        self.fshape = None
        self.kshape = None


    def im2c_setup(self,fshape,kshape):
        self.fshape = fshape
        self.kshape = kshape

        # For forward pass
        y,x = np.meshgrid(
            range(-self.ypad,fshape[1]+self.ypad-kshape[0]+1,self.stride),
            range(-self.xpad,fshape[2]+self.xpad-kshape[1]+1,self.stride),
            indexing='ij')
        oshape = (fshape[0],)+y.shape+(kshape[-1],)
        yd,xd = np.meshgrid(range(kshape[0]),range(kshape[1]),indexing='ij')
        y = y.reshape([-1,1,1])+yd.reshape([-1,1])
        x = x.reshape([-1,1,1])+xd.reshape([-1,1])
        fidx = np.reshape(range(fshape[0]),[-1,1,1,1])*fshape[1]
        fidx = ((fidx + y)*fshape[2] + x)*fshape[3] + range(fshape[3])
        fidx = fidx.reshape([fidx.shape[0]*fidx.shape[1],-1])
        vld = ((y >= 0) * (y < fshape[1]) * (x >= 0) * (x < fshape[2]))
        if not np.all(vld):
            vld = np.tile(vld[...,np.newaxis],[fshape[0],1,1,fshape[-1]]).reshape(-1)
        else:
            vld = None
        self.fidx = fidx
        self.vld = vld
        self.oshape = oshape

        # For backward pass
        if self.f.grad is None:
            return

        y,x = np.meshgrid(range(fshape[1]),range(fshape[2]),indexing='ij')
        yd,xd = np.meshgrid(range(kshape[0]),range(kshape[1]),indexing='ij')
        y = y.reshape([-1,1,1])-yd.reshape([-1,1])+self.ypad
        x = x.reshape([-1,1,1])-xd.reshape([-1,1])+self.xpad
        bfidx = np.reshape(range(fshape[0]),[-1,1,1,1])*oshape[1]
        bfidx = ((bfidx + y)*oshape[2] + x)*oshape[3] + range(oshape[3])
        bfidx = bfidx.reshape([bfidx.shape[0]*bfidx.shape[1],-1])
        bvld = ((y >= 0) * (y < oshape[1]) * (x >= 0) * (x < oshape[2]))
        if not np.all(bvld):
            bvld = np.tile(bvld[...,np.newaxis],[oshape[0],1,1,oshape[-1]]).reshape(-1)
        else:
            bvld = None
        self.bfidx = bfidx
        self.bvld = bvld

    def forward(self):
        fshape = self.f.value.shape
        kshape = self.k.value.shape
        if fshape != self.fshape or kshape != self.kshape:
            self.im2c_setup(fshape,kshape)

        fmat = _im2c(self.f.value,self.fidx,self.vld)
        kmat = self.k.value.reshape([-1,kshape[-1]])
        if self.k.grad is not None:
            self.fmat = fmat

        self.value = np.matmul(fmat,kmat).reshape(self.oshape)

    def backward(self):
        if self.f.grad is not None:
            gmat = _im2c(self.grad,self.bfidx,self.bvld)
            kmat = np.transpose(self.k.value,[0,1,3,2]).copy().reshape([-1,self.kshape[-2]])
            self.f.grad = self.f.grad + np.matmul(gmat,kmat).reshape(self.fshape)

        if self.k.grad is not None:
            kgrad = np.matmul(self.fmat.T,self.grad.reshape([-1,self.kshape[-1]]))
            self.k.grad = self.k.grad + kgrad.reshape(self.kshape)
            
'''
  Class name: ConvNaive
  Class usage: convolution layer given image feature f and filter k, stride and pad. this loop over the position in image and slightly lower than Conv. 
  Class function:
      forward: do concolution
      backward: calculate derivative w.r.t to f and filter k
'''            
class ConvNaive:

    def __init__(self,f,k,stride=1,pad=0):
        components.append(self)
        self.f = f
        self.k = k
        pad = np.array(pad)
        if pad.shape == ():
            self.xpad = self.ypad = pad
        else:
            self.ypad = pad[0]
            self.xpad = pad[1]

        self.stride=stride
        self.grad = None if f.grad is None and k.grad is None else DT(0)

    def forward(self):

        fshape = self.f.value.shape
        kshape = self.k.value.shape

        b = fshape[0]
        h = fshape[1]
        w = fshape[2]
        c1 = fshape[3]
        k = kshape[0]
        ch = kshape[3]

        self.value = np.zeros((b,np.int32((h-k+2*self.ypad)/self.stride+1), np.int32((w-k+2*self.xpad)/self.stride+1), ch))
        self.padf = np.zeros((b, h+2*self.ypad, w+2*self.xpad, c1))
        self.padf[:,self.ypad:h+self.ypad,self.xpad:w+self.xpad, :] = self.f.value

        # over positions in image 
        for y in range(np.int32((h-k+2*self.ypad)/self.stride + 1)):
            for x in range(np.int32((w-k+2*self.xpad)/self.stride + 1)):
                inx = self.padf[:,y*self.stride:y*self.stride+k,x*self.stride:x*self.stride+k,:].reshape((b, k*k*c1))
                ke = self.k.value.reshape((k*k*c1, ch))
                self.value[:,y,x,:] = np.matmul(inx, ke).reshape((b,ch))


    def backward(self):

        fshape = self.f.value.shape
        kshape = self.k.value.shape

        b = fshape[0]
        c1 = fshape[3]
        k = kshape[0]
        ch = kshape[3]

        h_hat = self.grad.shape[1]
        w_hat = self.grad.shape[2]
        h = (h_hat-1)*self.stride + k # padded image
        w = (w_hat-1)*self.stride + k # padded image
        fil_mid = k//2

        if self.f.grad is not None and self.k.grad is not None:
            fgrad = np.zeros((b, h, w, c1))
            kgrad = np.zeros((k, k, c1, ch))
            k_flip = np.transpose(self.k.value, (0,1,3,2))
            padf_flip = np.transpose(self.padf, (1,2,3,0))

            for y in range(h_hat):
                for x in range(w_hat):
                    out_grad_value = self.grad[:, y, x, :].reshape(b, ch)
                    y_img = y*self.stride; x_img = x*self.stride
                    fgrad[:, y_img:y_img+k, x_img:x_img+k, :] += np.dot(out_grad_value, k_flip).reshape(b,k,k,c1)
                    kgrad += np.dot(padf_flip[y_img:y_img+k, x_img:x_img+k,:,:], out_grad_value)

            self.f.grad = self.f.grad + fgrad[:, self.ypad:h-self.ypad,self.xpad:w-self.xpad, :]
            self.k.grad = self.k.grad + kgrad

'''
  Class name: MaxPool
  Class usage: max pooling layer. 
  Class function:
      forward: do max pooling given image feature and window size.
      backward: calculate derivative w.r.t to image feature x
'''            
class MaxPool:
    def __init__(self,x,ksz=2,stride=None):
        components.append(self)
        self.x = x
        self.ksz=ksz
        if stride is None:
            self.stride=ksz
        else:
            self.stride=stride
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        st = self.stride
        ksz = self.ksz
        self.value = -np.inf
        for y in range(ksz):
            for x in range(ksz):
                self.value = np.maximum(self.value,self.x.value[:,y::st,x::st,:])

    def backward(self):
        if self.x.grad is not None:
            st = self.stride
            ksz = self.ksz
            self.x.grad = self.x.grad + np.zeros_like(self.x.value)
            for y in range(ksz):
                for x in range(ksz):
                    self.x.grad[:,y::st,x::st,:] = self.grad * \
                            (self.value == self.x.value[:,y::st,x::st,:]) + \
                            self.x.grad[:,y::st,x::st,:]

'''
  Class name: AvePool
  Class usage: average pooling layer. 
  Class function:
      forward: do average pooling given image feature and window size.
      backward: calculate derivative w.r.t to image feature x
'''                            
class AvePool:
    def __init__(self,x,ksz=2,stride=None):
        components.append(self)
        self.x = x
        self.ksz=ksz
        if stride is None:
            self.stride=ksz
        else:
            self.stride=stride
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        st = self.stride
        ksz = self.ksz
        self.value = DT(0)
        for y in range(ksz):
            for x in range(ksz):
                self.value += self.x.value[:,y::st,x::st,:]
        self.value = self.value/ksz/ksz

    def backward(self):
        if self.x.grad is not None:
            st = self.stride
            ksz = self.ksz
            self.x.grad = self.x.grad + np.zeros_like(self.x.value)
            for y in range(ksz):
                for x in range(ksz):
                    self.x.grad[:,y::st,x::st,:] = self.grad/ksz/ksz + \
                            self.x.grad[:,y::st,x::st,:]

'''
  Class name: SumPool
  Class usage: summation pooling layer. 
  Class function:
      forward: do sum pooling given image feature and window size.
      backward: calculate derivative w.r.t to image feature x
'''                        
class SumPool:
    def __init__(self,x,ksz=2,stride=None):
        components.append(self)
        self.x = x
        self.ksz=ksz
        if stride is None:
            self.stride=ksz
        else:
            self.stride=stride
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        st = self.stride
        ksz = self.ksz
        self.value = DT(0)
        for y in range(ksz):
            for x in range(ksz):
                self.value += self.x.value[:,y::st,x::st,:]

    def backward(self):
        if self.x.grad is not None:
            st = self.stride
            ksz = self.ksz
            self.x.grad = self.x.grad + np.zeros_like(self.x.value)
            for y in range(ksz):
                for x in range(ksz):
                    self.x.grad[:,y::st,x::st,:] = self.grad + \
                            self.x.grad[:,y::st,x::st,:]

'''
  Class name: BatchNorm
  Class usage: Batch normalization layer. 
  Class function:
      forward:
      backward:
'''                        
class BatchNorm:

    def __init__(self, x, gamma, beta, test, ep=1e-30):

        components.append(self)
        self.x = x
        self.gamma = gamma
        self.beta = beta
        self.ep = ep
        self.grad = None if x.grad is None and beta.grad is None and gamma.grad is None else DT(0)

        # control test/train
        self.test = test
        self.batch = 0.
        self.mean = DT(0)
        self.var = DT(0)

    def forward(self):

        m = self.x.value.shape[0]
        
        # unbiased estimator
        if not self.test.value:
            self.batch_mean = np.mean(self.x.value,axis=0)
            self.batch_var = np.asarray(np.matrix(self.x.value).var(0)).reshape(-1)*DT(m)/(m-1.0)
            self.mean = (self.mean*self.batch + self.batch_mean)/(self.batch + 1)
            self.var =  (self.var*self.batch + self.batch_var)/(self.batch + 1)
            self.batch += 1
        else:
            self.batch_mean = self.mean
            self.batch_var = self.var
            self.batch = 0.

        # cached variable
        self.xhat = self.x.value - self.batch_mean
        self.var = self.batch_var + DT(self.ep)
        self.invar = DT(1.)/np.sqrt(self.var)
        self.value = self.xhat/self.invar*self.gamma.value + self.beta.value

    def backward(self):

        m = self.x.value.shape[0]
        dxhat = self.grad*self.gamma.value
        dvar = -0.5*np.sum(dxhat*self.xhat)*(self.var**(-1.5))
        dmean = np.sum(-dxhat*self.invar,axis=0) + (-2.0/m)*np.sum(self.xhat, axis=0)*dvar
        dx = dxhat*self.invar + (DT(2)/m)*self.xhat*dvar + dmean/DT(m)
        dgamma = np.sum(self.grad*self.xhat*self.invar, axis=0)
        dbeta =  np.sum(self.grad, axis=0)

        # backward the gradient
        if self.x.grad is not None:
            self.x.grad = self.x.grad + dx

        if self.beta.grad is not None:
            self.beta.grad = self.beta.grad + dbeta

        if self.gamma.grad is not None:
            self.gamma.grad = self.gamma.grad + dgamma


'''
  Class name: Dropout
  Class usage: Dropout layer. 
  Class function:
      forward:
      backward:
'''
class Dropout:
    def __init__(self,x,rate):
        components.append(self)
        self.x = x
        self.rate = rate
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        if self.rate.value == 0.:
            self.value = self.x.value
        else:
            self.mask = DT(np.random.rand(*self.x.value.shape) >= self.rate.value)
            self.value = self.x.value*self.mask/(1.-self.rate.value)

    def backward(self):
        if self.x.grad is not None:
            if self.rate.value == 0.:
                self.x.grad = self.grad
            else:
                self.x.grad = self.x.grad + self.grad * self.mask / (1.-self.rate.value)


def Momentum(lr,mom):
    if 'grad_hist' not in params[0].__dict__.keys():
        for p in params:
            p.grad_hist = DT(0)
    for p in params:
        p.grad_hist = mom*p.grad_hist + p.grad
        p.grad = p.grad_hist
    SGD(lr)

def AdaGrad(lr, ep=1e-8):
    if 'grad_G' not in params[0].__dict__.keys():
        for p in params:
              p.grad_G = DT(0)
    for p in params:
        p.grad_G = p.grad_G + p.grad*p.grad
        p.grad = p.grad/np.sqrt(p.grad_G + DT(ep))
    SGD(lr)

def RMSProp(lr, g=0.9, ep=1e-8):
    if 'grad_hist' not in params[0].__dict__.keys():
        for p in params:
              p.grad_hist = DT(0)
    for p in params:
        p.grad_hist = g*p.grad_hist + (1-g)*p.grad*p.grad
        p.grad = p.grad/np.sqrt(p.grad_hist + DT(ep))
    SGD(lr)

_a_b1t=DT(1.0)
_a_b2t=DT(1.0)
def Adam(alpha=0.001,b1=0.9,b2=0.999,ep=1e-8):
    global _a_b1t
    global _a_b2t

    if 'grad_hist' not in params[0].__dict__.keys():
        for p in params:
            p.grad_hist = DT(0)
            p.grad_h2 = DT(0)

    b1 = DT(b1)
    b2 = DT(b2)
    ep = DT(ep)
    _a_b1t = _a_b1t*b1
    _a_b2t = _a_b2t*b2
    for p in params:
        p.grad_hist = b1*p.grad_hist + (1.-b1)*p.grad
        p.grad_h2 = b2*p.grad_h2 + (1.-b2)*p.grad*p.grad

        mhat = p.grad_hist / (1. - _a_b1t)
        vhat = p.grad_h2 / (1. - _a_b2t)

        p.grad = mhat / (np.sqrt(vhat) + ep)
    SGD(alpha)

# clip the gradient if the norm of gradient is larger than some threshold, this is crucial for RNN.     
def GradClip(grad_clip):
    for p in params:
        l2 = np.sqrt(np.sum(p.grad*p.grad))
        if l2 >= grad_clip:
            p.grad *= grad_clip/l2

            
##################################################### Recurrent Components ##############################################

'''
  Class name: Embed
  Class usage: Embed layer. 
  Class function:
      forward: given the embeeding matrix w2v and word idx, return its corresponding embedding vector. 
      backward: calculate the derivative w.r.t to embedding matrix
'''
class Embed:

    def __init__(self,idx,w2v):
        components.append(self)
        self.idx = idx
        self.w2v = w2v
        self.grad = None if w2v.grad is None else DT(0)

    def forward(self):
        self.value = self.w2v.value[np.int32(self.idx.value),:]
    def backward(self):
        
        if self.w2v.grad is not None:
            if isinstance(self.w2v.grad, np.float32):
                self.w2v.grad = np.zeros(self.w2v.value.shape)
                #print ("initialize")
            else:
                #print ("update")
                self.w2v.grad[np.int32(self.idx.value),:] = self.w2v.grad[np.int32(self.idx.value),:] + self.grad
            
'''
  Class name: ConCat
  Class usage: ConCat layer. 
  Class function:
      forward: concat two matrix along with the axis 1.
      backward: calculate the derivative w.r.t to matrix a and y.
'''            
class ConCat:

    def __init__(self, x, y):
        components.append(self)
        self.x = x 
        self.y = y
        self.grad = None if x.grad is None and y.grad is None else DT(0)
       
    def forward(self):
        
        self.value = np.concatenate((self.x.value, self.y.value), axis=1)
        
    def backward(self):     
        
        dim_x = self.x.value.shape[1]
        dim_y = self.y.value.shape[1]
        
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad[:, 0:dim_x]
        if self.y.grad is not None:
            self.y.grad = self.y.grad + self.grad[:, dim_x:dim_x+dim_y]

'''
  Class name: ArgMax
  Class usage: ArgMax layer. 
  Class function:
      forward: given x, calculate the index which has the maximum value 
      backward: None
'''            
class ArgMax:

    def __init__(self, x):
        components.append(self)
        self.x = x 
  
    def forward(self):
        self.value = np.argmax(self.x.value)

    def backward(self):     
        pass
