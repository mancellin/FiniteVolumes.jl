#!/usr/bin/env julia

using LinearAlgebra: norm
using FiniteVolumes

grid = PeriodicRegularMesh2D(50, 50)
model = directional_splitting(ScalarLinearAdvection([1.0, 1.0]))

is_in_disk(i) = norm([0.5, 0.5] - cell_center(grid, i)) < 0.3
is_in_square(i, a=0.5) = all(0.5-a/2 .<= cell_center(grid, i) .<= 0.5+a/2)

dt = 0.004
nb_period = 1.0
nb_time_steps = ceil(Int, nb_period/dt)

w₀ = [is_in_square(i) ? 1.0 : 0.0 for i in 1:nb_cells(grid)]

t, w_upwind = FiniteVolumes.run(model, grid, w₀, dt=dt, nb_time_steps=nb_time_steps)

t, w_minmod = FiniteVolumes.run(model, grid, w₀, dt=dt, nb_time_steps=nb_time_steps,
								numerical_flux=Muscl(limiter=minmod))

t, w_ultra = FiniteVolumes.run(model, grid, w₀, dt=dt, nb_time_steps=nb_time_steps,
							   numerical_flux=Muscl(limiter=ultrabee))

t, w_lagout = FiniteVolumes.run(model, grid, w₀, dt=dt, nb_time_steps=nb_time_steps,
                                numerical_flux=LagoutiereDownwind())

using Plots
plot(
     plot(grid, w₀, 1, title="exact"),
     plot(grid, w_upwind, 1, title="upwind"),
     plot(grid, w_minmod, 1, title="minmod"),
     plot(grid, w_ultra, 1, title="ultrabee"),
     plot(grid, w_lagout, 1, title="lagoutiere"),
    )
