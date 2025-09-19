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

import numpy as xp
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
		xp.random.seed(seed)
		self.phases = 2 * xp.pi * xp.random.random((self.M, self.M))
		self.nm = xp.meshgrid(xp.arange(self.M+1), xp.arange(self.M+1), indexing='ij')
		self.phic = xp.zeros((self.M+1, self.M+1), dtype=xp.complex128)
		self.phic[1:, 1:] = self.A / (self.nm[0][1:, 1:]**2 + self.nm[1][1:, 1:]**2)**1.5 * xp.exp(1j * self.phases)
		sqrt_nm = xp.sqrt(self.nm[0]**2 + self.nm[1]**2)
		self.phic[sqrt_nm > self.M] = 0
		self.d1phic = xp.asarray([-self.nm[1] * self.phic, self.nm[0] * self.phic])	
		self.d2phic = xp.asarray([-self.nm[0]**2 * self.phic, -self.nm[0] * self.nm[1] * self.phic,\
							  -self.nm[1]**2 * self.phic])

	def initial_conditions(self, n_traj=1, x=None, y=None, type='fixed', seed=None):
		x, y = (0, 2 * xp.pi) if x is None else x, (0, 2 * xp.pi) if y is None else y
		if type == 'random':
			seed = seed if seed is not None else int(time.time()) + os.getpid()
			rng = xp.random.default_rng(seed)
			x0 = (x[-1] - x[0]) * rng.random(n_traj) + x[0]
			y0 = (y[-1] - y[0]) * rng.random(n_traj) + y[0]
			z0 = xp.concatenate((x0, y0), axis=None)
		elif type == 'fixed':
			n_traj = int(xp.sqrt(n_traj))**2
			x0 = xp.linspace(x[0], x[-1], int(xp.sqrt(n_traj)), endpoint=False)
			y0 = xp.linspace(y[0], y[-1], int(xp.sqrt(n_traj)), endpoint=False)
			x0, y0 = xp.meshgrid(x0, y0, indexing='ij')
			z0 = xp.concatenate((x0.flatten(), y0.flatten()), axis=None)
		return z0
	
	def y_dot(self, t, z):
		exp_xy = xp.exp(1j * (xp.einsum('ijk,i...->jk...', self.nm, xp.split(z, 2)) - t))
		return (xp.einsum('ijk,jk...->i...', self.d1phic, exp_xy).real).reshape(z.shape)
	
	def k_dot(self, t, z):
		exp_xy = xp.exp(1j * (xp.einsum('ijk,i...->jk...', self.nm, xp.split(z, 2)) - t))
		return xp.sum(xp.einsum('jk,jk...->...', self.phic, exp_xy).real)
	
	def potential(self, t, z, dx=0, dy=0):
		exp_xy = xp.exp(1j * (xp.einsum('ijk,i...->jk...', self.nm, xp.split(z, 2)) - t))
		cases = {
			(0, 0): (self.phic, 'imag'),
			(1, 0): (self.d1phic[1], 'real'),
			(0, 1): (-self.d1phic[0], 'real'),
			(2, 0): (self.d2phic[0], 'imag'),
			(1, 1): (self.d2phic[1], 'imag'),
			(0, 2): (self.d2phic[2], 'imag'),
		}
		coeff, part = cases.get((dx, dy), (None, None))
		if coeff is None:
			raise ValueError("Only first and second derivatives are implemented")
		result = xp.einsum('ijk,jk...->...', coeff, exp_xy)
		return getattr(result, part)
	
	def hamiltonian(self, t, z):
		return xp.sum(self.potential(t, z))
	
	def y_dot_lyap(self, t, z):
		x, y, J11, J12, J21, J22 = xp.split(z, 6)
		z_dot = self.y_dot(t, xp.concatenate((x, y), axis=None))
		exp_xy = xp.exp(1j * (xp.einsum('ijk,i...->jk...', self.nm, (x, y)) - t))
		d2phi = xp.einsum('ijk,jk...->i...', self.d2phic, exp_xy).imag
		J11_dot = -J11 * d2phi[1] - J21 * d2phi[2]
		J12_dot = -J12 * d2phi[1] - J22 * d2phi[2]
		J21_dot = J11 * d2phi[0] + J21 * d2phi[1]
		J22_dot = J12 * d2phi[0] + J22 * d2phi[1]
		return xp.concatenate((z_dot, J11_dot, J12_dot, J21_dot, J22_dot), axis=None)
	
	def plot_sol(self, sol, wrap=False): 
		x, y = xp.split(sol.y, 2)
		if wrap:
			x, y = xp.asarray(x) % (2 * xp.pi), xp.asarray(y) % (2 * xp.pi)
		plt.plot(x, y, '.', color='blue')
		plt.xlabel('x')
		plt.ylabel('y')
		if wrap:
			plt.xlim(0, 2 * xp.pi)
			plt.ylim(0, 2 * xp.pi)
		plt.show()
