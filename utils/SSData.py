from scipy.io import loadmat
import torch
import numpy as np

class SSData:

	def __init__(self, mat_file):
		self.mat_file = mat_file
		data = loadmat(mat_file)
		
		# Unpacks matlab file and returns torch tensors
		out = dict()
		good_keys = list(data.keys())[3:]
		for key in good_keys:
			out[key] = torch.tensor(data[key].astype(np.float64)).type(torch.double)
		
		self.A = out['A']
		if self.A.dim() < 3 and self.A.size()[-1] != 1:
			self.A = self.A.reshape(int(out['N'].item()),-1,1)
		self.n_x = self.A.size()[-1]

		self.B = out['B']
		if self.B.dim() < 3 and self.B.size()[-1] != 1:
			self.B = self.B.reshape(int(out['N'].item()),-1,1)
		self.n_u = self.B.size()[-1]
		
		self.C = out['C']
		if self.C.dim() < 3 and self.C.size()[-1] != 1:
			self.C = self.C.reshape(int(out['N'].item()),-1,1)
		self.n_y = self.C.size()[-2]
		
		self.D = out['D']
		if self.D.dim() < 3 and self.D.size()[-1] != 1:
			self.D = self.D.reshape(int(out['N'].item()),-1,1)

		self.N = int( out['N'].item() )
		self.dT = out['dT'].item()

	def getSSmatrices(self):
		if self.N == 1:
			return self.A, self.B, self.C, self.D
		else:
			return self.A[0:1,:,:], self.B[0:1,:,:], self.C[0:1,:,:], self.D[0:1,:,:]
