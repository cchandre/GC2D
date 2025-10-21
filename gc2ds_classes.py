#
# BSD 2-Clause License
#
# Copyright (c) 2025, Cristel Chandre
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from pyhamsys import HamSys
import matplotlib.pyplot as plt
import os
import time

class GC2Ds(HamSys):
	def __str__(self) -> str:
		return f'2D Guiding Center ({self.__class__.__name__}) for turbulent potentials'

	def __init__(self, params) -> None:
		super().__init__(ndof=1.5)
		self.A, self.M = params["A"], params["M"]
		seed = params["seed"] if "seed" in params else 27
		np.random.seed(seed)
		self.phases = 2 * np.pi * np.random.random((self.M, self.M))
		self.nm = np.meshgrid(np.arange(self.M+1), np.arange(self.M+1), indexing='ij')
		self.phic = np.zeros((self.M+1, self.M+1), dtype=np.complex128)
		self.phic[1:, 1:] = self.A / (self.nm[0][1:, 1:]**2 + self.nm[1][1:, 1:]**2)**1.5 * np.exp(1j * self.phases)
		sqrt_nm = np.sqrt(self.nm[0]**2 + self.nm[1]**2)
		self.phic[sqrt_nm > self.M] = 0
		self.d1phic = np.asarray([self.nm[0] * self.phic, self.nm[1] * self.phic])
		self.d2phic = np.asarray([-self.nm[0]**2 * self.phic, -self.nm[0] * self.nm[1] * self.phic,
							  -self.nm[1]**2 * self.phic])

	def initial_conditions(self, n_traj=1, x=None, y=None, kind='fixed', seed=None):
		x, y = (0, 2 * np.pi) if x is None else x, (0, 2 * np.pi) if y is None else y
		if kind == 'random':
			seed = seed if seed is not None else int(time.time()) + os.getpid()
			rng = np.random.default_rng(seed)
			x0, y0 = rng.uniform(x[0], x[-1], n_traj), rng.uniform(y[0], y[-1], n_traj)
		elif kind == 'fixed':
			n_side = int(np.sqrt(n_traj))
			x0 = np.linspace(x[0], x[-1], n_side, endpoint=False)
			y0 = np.linspace(y[0], y[-1], n_side, endpoint=False)
			x0, y0 = np.meshgrid(x0, y0, indexing='ij')
		else:
			raise ValueError("Invalid 'kind' argument. Must be 'fixed' or 'random'.")
		return np.concatenate((x0.ravel(), y0.ravel()), axis=None)
	
	def compute_exp(self, t, z):
		return np.exp(1j * (np.einsum('ijk,i...->jk...', self.nm, np.split(z, 2), optimize=True) - t))

	def y_dot(self, t, z):
		exp_xy = self.compute_exp(t, z)
		d1phi = np.einsum('ijk,jk...->i...', self.d1phic, exp_xy).real
		return np.concatenate((-d1phi[1], d1phi[0]), axis=None)
	
	def k_dot(self, t, z):
		exp_xy = self.compute_exp(t, z)
		return np.sum(np.einsum('jk,jk...->...', self.phic, exp_xy).real)

	def potential(self, t, z, dx=0, dy=0):
		exp_xy = self.compute_exp(t, z)
		cases = {
			(0, 0): (self.phic, 'imag'),
			(1, 0): (self.d1phic[0], 'real'),
			(0, 1): (self.d1phic[1], 'real'),
			(2, 0): (self.d2phic[0], 'imag'),
			(1, 1): (self.d2phic[1], 'imag'),
			(0, 2): (self.d2phic[2], 'imag')}
		coeff, part = cases.get((dx, dy), (None, None))
		if coeff is None:
			raise ValueError("Only first and second derivatives are implemented")
		result = np.einsum('jk,jk...->...', coeff, exp_xy)
		return getattr(result, part)
	
	def hamiltonian(self, t, z):
		return np.sum(self.potential(t, z))

	def y_dot_lyap(self, t, z):
		x, y, *J = np.split(z, 6)
		z = np.concatenate((x, y), axis=None)
		z_dot = self.y_dot(t, z)
		J = np.array(J).reshape((2, 2, -1))
		exp_xy = self.compute_exp(t, z)
		d2phi = np.einsum('ijk,jk...->i...', self.d2phic, exp_xy).imag
		A = np.array([[-d2phi[1], -d2phi[2]], [d2phi[0], d2phi[1]]])
		J_dot = np.einsum('ijm,jkm->ikm', A, J)
		return np.concatenate((z_dot, J_dot.reshape(-1)), axis=None)
	
	def plot_sol(self, sol, wrap=False): 
		x, y = np.split(sol.y, 2)
		if wrap:
			x, y = np.asarray(x) % (2 * np.pi), np.asarray(y) % (2 * np.pi)
		plt.plot(x.T, y.T, '.')
		plt.xlabel('x')
		plt.ylabel('y')
		if wrap:
			plt.xlim(0, 2 * np.pi)
			plt.ylim(0, 2 * np.pi)
		plt.show()
