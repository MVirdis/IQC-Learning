import numpy as np
import torch
import torch.nn as nn

def t(x):
	# Alias for transpose
	if type(x) == int:
		return x
	return torch.transpose(x,x.dim()-2,x.dim()-1)

class REN(nn.Module):
	"""
	Implementation of an acyclic R-REN [2] without feedthrough (i.e. D_22 = 0).
	"""

	def __init__(self, n_x, n_units, n_y, n_u, Q, S, R, bias=False, train_qsr=False):
		super(REN, self).__init__()
		self.n_x = n_x
		self.n_y = n_y
		self.n_u = n_u
		self.n_units = n_units
		
		# Free parameters
		self.B_2 = nn.Parameter(torch.randn(n_x,n_u))
		self.C_2 = nn.Parameter(torch.randn(n_y,n_x))
		self.D_12 = nn.Parameter(torch.randn(n_units,n_u))
		self.D_21 = nn.Parameter(torch.randn(n_y,n_units))
		if bias:
			self.b = nn.Parameter(torch.randn(n_x+n_units+n_y,1))
		else:
			self.b = torch.zeros(n_x+n_units+n_y,1)
		
		# Free matrices
		self.X = nn.Parameter(torch.randn(2*n_x+n_units,2*n_x+n_units))
		self.Y1 = nn.Parameter(torch.randn(n_x, n_x))
		
		# IQC Multiplier
		self.train_qsr = train_qsr
		if not train_qsr:
			self.Q = Q
			self.S = t(S)
			self.R = R
			self.Rinv = torch.linalg.solve(R,torch.eye(self.n_u))
			#print(self.Rinv)
		else:
			self.L_Q = nn.Parameter(torch.randn(n_y,n_y)) + 1e-6*torch.eye(n_y)
			self.L_R = nn.Parameter(torch.randn(n_u,n_u)) + 1e-6*torch.eye(n_y)
			self.S = nn.Parameter(torch.randn(n_u,n_y))

		# Check inertia of multiplier
		eigvs = torch.real(torch.linalg.eigvals(Q))
		torch._assert(torch.all(eigvs <= 0.0).item(), "Q must be negative semi-definite")
		eigvs = torch.real(torch.linalg.eigvals(R))
		torch._assert(torch.all(eigvs > 0.0).item(), "R must be positive definite")

		# Activation function
		# self.sigma = torch.nn.LeakyReLU(0.3)
		# self.sigma = torch.nn.Sigmoid()
		self.sigma = torch.nn.Tanh()
		
		self.updateConstrainedWeights()
		
	def updateConstrainedWeights(self):
		# Compute and update the constrained weights

		if self.train_qsr:
			self.Q = -torch.matmul(self.L_Q, self.L_Q)
			self.Rinv = torch.matmul(self.L_R, self.L_R)
		
		C2_ = torch.matmul(self.S, self.C_2)
		torch._assert(C2_.size() == (self.n_u, self.n_x),
					  "Mismatch C2_ size")

		D21_ = torch.matmul(self.S, self.D_21) - t(self.D_12)
		torch._assert(D21_.size() == (self.n_u, self.n_units),
					  "Mismatch D21_ size")
		
		H = torch.matmul(t(self.X),self.X)
		torch._assert(H.size() == (2*self.n_x+self.n_units, 2*self.n_x+self.n_units),
					  "Mismatch H size")
		torch._assert(torch.eye(self.X.size()[0]).size() == (2*self.n_x+self.n_units,2*self.n_x+self.n_units),
					  "Mismatch I size")
		H = H + 1e-3*torch.eye(2*self.n_x + self.n_units)
		lTerm1 = torch.cat( (t(C2_), t(D21_), self.B_2) )
		torch._assert(lTerm1.size() == (self.n_x+self.n_units+self.n_x, self.n_u),
					  "Mismatch lTerm1 size")
		H = H + torch.matmul(
			torch.matmul( lTerm1, self.Rinv),
			t(lTerm1)
		)

		lTerm2 = torch.cat((
					t(self.C_2),
					t(self.D_21),
					torch.zeros(self.n_x,self.n_y)
					), dim=0)
		torch._assert(lTerm2.size() == (self.n_x+self.n_units+self.n_x, self.n_y),
					  "Mismatch lTerm2 size")
		H = H - torch.matmul(
			torch.matmul(lTerm2, self.Q),
			t(lTerm2)
		)
		
		self.H = H

		H_11 = H[0:self.n_x, 0:self.n_x]
		H_12 = H[0:self.n_x, self.n_x:self.n_x+self.n_units]
		H_13 = H[0:self.n_x, self.n_x+self.n_units:]

		H_21 = H[self.n_x:self.n_x+self.n_units, 0:self.n_x]
		H_22 = H[self.n_x:self.n_x+self.n_units, self.n_x:self.n_x+self.n_units]
		H_23 = H[self.n_x:self.n_x+self.n_units, self.n_x+self.n_units:]

		H_31 = H[self.n_x+self.n_units:, 0:self.n_x]
		H_32 = H[self.n_x+self.n_units:, self.n_x:self.n_x+self.n_units]
		H_33 = H[self.n_x+self.n_units:, self.n_x+self.n_units:]

		H_test = torch.cat((torch.cat((H_11, H_12, H_13), dim=1),
					 		torch.cat((H_21, H_22, H_23), dim=1),
					 		torch.cat((H_31, H_32, H_33), dim=1)), dim=0)
		torch._assert(torch.all(H_test == H).item(), "H partitioning is wrong")
		
		# Compute constrained weights and update attributes
		self.F = H_31
		self.B_1 = H_32
		self.C_1 = -H_21
		self.P = H_33
		alpha = 1
		self.E = 0.5*(H_11 + 1/(alpha**2)*self.P + self.Y1 -t(self.Y1))
		torch._assert(torch.diag(H_22).size() == (self.n_units,),
					  "Wrong diagonal extraction from H_22")
		PHI = torch.diag(torch.diag(H_22))
		torch._assert(PHI.size() == (self.n_units, self.n_units),
					  "Wrong PHI size")
		L = torch.tril(-(H_22-PHI))
		torch._assert(L.size() == (self.n_units, self.n_units),
					  "Wrong L size")
		self.Lambda = 0.5*PHI
		self.D_11 = L

		# self.Einv = torch.inverse(self.E)
		# torch._assert(self.Einv.size() == (self.n_x, self.n_x),
		# 			  "Wrong Einv size")
		# self.Lambdainv = torch.diag(1/torch.diag(self.Lambda))
		# torch._assert(self.Lambdainv.size() == (self.n_units, self.n_units),
		# 			  "Wrong Lambdainv size")

		# # Check how accurate the inverses are
		# dE = torch.matmul(self.Einv.detach(), self.E.detach())-torch.eye(self.n_x, self.n_x)
		# dE_ = torch.trace(torch.matmul(t(dE), dE)).item()
		# torch._assert(dE_ < 1e-6, "Large dE error {}".format(dE_))

		# dLambda = torch.matmul(self.Lambdainv.detach(), self.Lambda.detach())-torch.eye(self.n_units, self.n_units)
		# dLambda_ = torch.trace(torch.matmul(t(dLambda), dLambda)).item()
		# torch._assert(dLambda_ < 1e-6, "Large dLambda error {}".format(dLambda_))
	
	def getExplicitSS(self):
		# Returns the explicit state-space model
		A = torch.linalg.solve(self.E, self.F)
		B1 = torch.linalg.solve(self.E, self.B_1)
		B2 = torch.linalg.solve(self.E, self.B_2)
		C1 = torch.linalg.solve(self.Lambda, self.C_1)
		D11 = torch.linalg.solve(self.Lambda, self.D_11)
		D12 = torch.linalg.solve(self.Lambda, self.D_12)
		C2 = self.C_2
		D21 = self.D_21
		# D22 = self.D_22
		
		# return {'A': A.detach().numpy(),'B1': B1.detach().numpy(),'B2': B2.detach().numpy(),'C1':C1.detach().numpy(),'D11':D11.detach().numpy(),'D12':D12.detach().numpy(),'C2': C2.detach().numpy(),'D21':D21.detach().numpy(),'D22':D22.detach().numpy()}
		return {'A': A.detach().numpy().astype('double'),
				'B1': B1.detach().numpy().astype('double'),
				'B2': B2.detach().numpy().astype('double'),
				'C1':C1.detach().numpy().astype('double'),
				'D11':D11.detach().numpy().astype('double'),
				'D12':D12.detach().numpy().astype('double'),
				'C2': C2.detach().numpy().astype('double'),
				'D21':D21.detach().numpy().astype('double')}
	
	def forward(self, u, x0=None, get_state=False):
		torch._assert(u.dim() == 3, "u has to have 3 dimensions (batch_size,n_u,1)")
		torch._assert(u.size()[1] == self.n_u, "u has wrong size {}".format(u.size()[1]))

		if x0 is not None:
			torch._assert(x0.dim() == 3, "x0 has to have 3 dimensions (batch_size,n_x,1)")
			torch._assert(x0.size()[1:] == (self.n_x,1), "x0 has to have 3 dimensions (batch_size,n_x,1)")

			torch._assert(x0.size()[0] == u.size()[0], "x0 has to have the same batch_size of u")
		
		batch_size = u.size(dim=0)
		b_ = torch.reshape(self.b, (1, self.n_x+self.n_units+self.n_y, 1))

		# E_inv = self.Einv
		# Lambda_inv = self.Lambdainv
		
		# Compute output
		x = x0 if x0 is not None else torch.zeros(batch_size,self.n_x,1)
		y = torch.zeros(batch_size,self.n_y,1)
		
		w = torch.empty(batch_size,0,1)
		for i_unit in range(0,self.n_units):

			term1 = torch.matmul(self.C_1[i_unit:i_unit+1,:], x)
			torch._assert(self.C_1[i_unit:i_unit+1,:].size() == (1,self.n_x), "Wrong C_1 slicing")
			torch._assert(term1.size() == (batch_size,1,1), "term1 has wrong size")
			# print('Shape term1',term1.size())

			if i_unit > 0:
				term2 = torch.matmul(self.D_11[i_unit-1:i_unit,0:i_unit], w)
				torch._assert(term2.size() == (batch_size,1,1), "term2 has wrong size")
				# print('Shape term2',term2.size())
			else:
				term2 = torch.zeros(batch_size,1,1)

			term3 = torch.matmul(self.D_12[i_unit:i_unit+1,:], u)
			torch._assert(self.D_12[i_unit:i_unit+1,:].size() == (1,self.n_u), "Wrong D_12 slicing")
			torch._assert(term3.size() == (batch_size,1,1), "term3 has wrong size")
			# print('Shape term3',term3.size())

			term4 = b_[:,self.n_x+i_unit:self.n_x+i_unit+1,0:1]
			# print('Shape term4',term4.size())
			# print('Shape', torch.sigmoid(Lambda_inv[i_unit:i_unit+1,i_unit:i_unit+1]*(term1 + term2 + term3 + term4)).size())

			w_i = self.sigma(1/self.Lambda[i_unit:i_unit+1,i_unit:i_unit+1]*(term1 + term2 + term3 + term4))

			w = torch.cat(
				(w, w_i),
				dim=1
			)
		
		y_next = torch.matmul(self.C_2, x) + torch.matmul(self.D_21,w) + b_[:,self.n_x+self.n_units:,0:1]
		torch._assert(y_next.size() == (batch_size,self.n_y,1), "Wrong y_next size")
		
		if get_state:
			term1 = torch.matmul(self.F,x)
			torch._assert(term1.size() == (batch_size,self.n_x,1), "Wrong term1 x size")
			term2 = torch.matmul(self.B_1,w)
			torch._assert(term2.size() == (batch_size,self.n_x,1), "Wrong term2 x size")
			term3 = torch.matmul(self.B_2,u)
			torch._assert(term3.size() == (batch_size,self.n_x,1), "Wrong term3 x size")
			term4 = b_[:,0:self.n_x,0:1]
			torch._assert(term4.size() == (1,self.n_x,1), "Wrong b_x size")
			
			x_next = torch.linalg.solve(self.E, ( term1 + term2 + term3 + term4 ))
			
			return y_next, x_next
		else:
			return y_next

	def clone(self):
		phi = REN(self.n_x, self.n_units, self.n_y, self.n_u, self.Q, self.S, self.R, self.bias, self.train_qsr)
		
		# Free parameters
		phi.B_2 = torch.clone(self.B_2)
		phi.C_2 = torch.clone(self.C_2)
		phi.D_12 = torch.clone(self.D_12)
		phi.D_21 = torch.clone(self.D_21)
		phi.b = torch.clone(self.b)
		
		# Free matrices
		phi.X = torch.clone(self.X)
		phi.Y1 = torch.clone(self.Y1)
		phi.updateConstrainedWeights()

		return phi

	def checkLMI(self):
		return torch.all(torch.real(torch.linalg.eigvals(self.H)) > 0).item()
