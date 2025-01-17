# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport exp
import numpy as np

def sig_kernel_batch_varpar(double[:,:,:] G_static, bint _naive_solver=False):

	cdef int A = G_static.shape[0]
	cdef int M = G_static.shape[1]
	cdef int N = G_static.shape[2]
	cdef int i, j, l
	
	cdef double[:,:,:] K = np.zeros((A,M+1,N+1), dtype=np.float64)
		
	for l in range(A):

		for i in range(M+1):
			K[l,i,0] = 1.
	
		for j in range(N+1):
			K[l,0,j] = 1.

		for i in range(M):
			for j in range(N):

				if _naive_solver:
					K[l,i+1,j+1] = K[l,i+1,j] + K[l,i,j+1] + K[l,i,j]*(G_static[l,i,j] - 1.)
				else:
					K[l,i+1,j+1] = (K[l,i+1,j] + K[l,i,j+1])*(1. + 0.5*G_static[l,i,j]+(1./12)*G_static[l,i,j]**2) - K[l,i,j]*(1. - (1./12)*G_static[l,i,j]**2)
					#K[l,i+1,j+1] = K[l,i+1,j] + K[l,i,j+1] - K[l,i,j] + (exp(0.5*G_static[l,i,j])-1.)*(K[l,i+1,j] + K[l,i,j+1])

	return np.array(K)


def sig_kernel_Gram_varpar(double[:,:,:,:] G_static, bint sym=False, bint _naive_solver=False):

	cdef int A = G_static.shape[0]
	cdef int B = G_static.shape[1]
	cdef int M = G_static.shape[2]
	cdef int N = G_static.shape[3]
	cdef int i, j, l, m

	cdef double[:,:,:,:] K = np.zeros((A,B,M+1,N+1), dtype=np.float64)

	if sym:
		# for l in prange(A,nogil=True):
		for l in range(A):
			for m in range(l,A):

				for i in range(M+1):
					K[l,m,i,0] = 1.
					K[m,l,i,0] = 1.
	
				for j in range(N+1):
					K[l,m,0,j] = 1.
					K[m,l,0,j] = 1.

				for i in range(M):
					for j in range(N):

						if _naive_solver:
							K[l,m,i+1,j+1] = K[l,m,i+1,j] + K[l,m,i,j+1] + K[l,m,i,j]*(G_static[l,m,i,j]-1.)
						else:
							K[l,m,i+1,j+1] = (K[l,m,i+1,j] + K[l,m,i,j+1])*(1.+0.5*G_static[l,m,i,j]+(1./12)*G_static[l,m,i,j]**2) - K[l,m,i,j]*(1.-(1./12)*G_static[l,m,i,j]**2)
							#K[l,m,i+1,j+1] = K[l,m,i+1,j] + K[l,m,i,j+1] - K[l,m,i,j] + (exp(0.5*G_static[l,m,i,j])-1.)*(K[l,m,i+1,j] + K[l,m,i,j+1])

						K[m,l,j+1,i+1] = K[l,m,i+1,j+1]

	else:
		# for l in prange(A,nogil=True):
		for l in range(A):
			for m in range(B):

				for i in range(M+1):
					K[l,m,i,0] = 1.
	
				for j in range(N+1):
					K[l,m,0,j] = 1.

				for i in range(M):
					for j in range(N):

						if _naive_solver:
							K[l,m,i+1,j+1] = K[l,m,i+1,j] + K[l,m,i,j+1] + K[l,m,i,j]*(G_static[l,m,i,j] - 1.)
						else:
							K[l,m,i+1,j+1] = (K[l,m,i+1,j] + K[l,m,i,j+1])*(1. + 0.5*G_static[l,m,i,j]+(1./12)*G_static[l,m,i,j]**2) - K[l,m,i,j]*(1. - (1./12)*G_static[l,m,i,j]**2)
							#K[l,m,i+1,j+1] = K[l,m,i+1,j] + K[l,m,i,j+1] - K[l,m,i,j] + (exp(0.5*G_static[l,m,i,j])-1.)*(K[l,m,i+1,j] + K[l,m,i,j+1])
	
	return np.array(K)


def sig_kernel_Gram_varpar_const(double[:,:,:,:] G_static, double _lambda, bint sym=False, bint _naive_solver=False):
	cdef int A = G_static.shape[0]
	cdef int B = G_static.shape[1]
	cdef int M = G_static.shape[2]
	cdef int N = G_static.shape[3]
	cdef int i, j, l, m
	cdef double _aux
	cdef double[:,:,:,:] K = np.zeros((A,B,M+1,N+1), dtype=np.float64)

	for l in range(A):
		for m in range(B):

			for i in range(M+1):
				K[l,m,i,0] = 1.
				# K[l,m,i,1] = 1.

			for j in range(N+1):
				K[l,m,0,j] = 1.
				# K[l,m,1,j] = 1.
			# raise ValueError()
			for i in range(M):
				for j in range(N):
					_aux = G_static[l,m,i,j] * _lambda - 1.
					# fixme: central difference performance is worse
					# K[l,m,i+1,j+1] = K[l,m,i+1,j-1] + K[l,m,i-1,j+1] - K[l,m,i-1,j-1] + 4 * _aux * K[l,m,i,j]

					# forward difference

					K[l, m, i + 1, j + 1] = K[l, m, i + 1, j] + K[l, m, i, j + 1] + K[l, m, i, j] * _aux

					# fixme npn-naive implementation (no clue where it's coming from)
					# K[l, m, i + 1, j + 1] = \
					# 	(K[l, m, i + 1, j] + K[l, m, i, j + 1]) * (
					# 			1. + 0.5 * _aux + (1. / 12) * _aux ** 2
					# 	) - K[l, m, i, j] * (1. - (1. / 12) * _aux ** 2)

	return np.array(K)