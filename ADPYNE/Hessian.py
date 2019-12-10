import numpy as np

class Hessian():
	def __init__(self, f):
		self.value = f.Real.Real.Real

		fd1 = f.Dual[0].Real.Real[0].Real.Real
		fd2 = f.Dual[1].Real.Real[1].Real.Real
		self.firstDer = np.array([fd1,fd2])
		
		hxx = f.Dual[0].Dual[0].Real[0].Real
		hyy = f.Dual[0].Real.Real[1].Dual[0].Real[1].Real
		hxy = f.Dual[0].Dual[0].Real[1].Real
		hyx = hxy
		self.hessian = np.array([[hxx,hxy],[hyx,hyy]])


	def __str__(self):
		return "{} val\n\n{} der\n\n{} hess".format(self.value,self.firstDer,self.hessian)