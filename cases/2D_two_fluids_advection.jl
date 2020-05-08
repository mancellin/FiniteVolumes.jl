#!/usr/bin/env julia
"""Diagonal advection of a disk of liquid into gas"""

using LinearAlgebra: norm
using FiniteVolumes

grid = PeriodicRegularMesh2D(50, 50)
model = IsothermalTwoFluidEuler{nb_dims(grid)}(300.0, 1.0, 1500.0, 1000.0, 1e5)

is_in_disk(i) = norm([0.5, 0.5] - cell_center(grid, i)) < 0.3

w₀ = [full_state(model, p=1e5, ux=10.0, uy=10.0, ξ=(is_in_disk(i) ? 0.0 : 1.0)) for i in 1:nb_cells(grid)]
t_final, w = FiniteVolumes.run(model, grid, w₀, dt=3e-6, nb_time_steps=1_000)

using Plots
plot(plot(grid, w, :p), plot(grid, w, :α))

