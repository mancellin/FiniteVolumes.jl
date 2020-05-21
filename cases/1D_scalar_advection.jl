#!/usr/bin/env julia

using FiniteVolumes

grid = RegularMesh1D(0.0, 1.0, 100)
model = ScalarLinearAdvection(1.0)

w₀ = [i < nb_cells(grid)/2 ? 1.0 : 0.0 for i in 1:nb_cells(grid)]

t_final, w = FiniteVolumes.run(model, grid, w₀, cfl=0.2, nb_time_steps=100)

t_final, w_muscl = FiniteVolumes.run(model, grid, w₀, cfl=0.2, nb_time_steps=100,
									 numerical_flux=Muscl(limiter=minmod))

t_final, w_ultra = FiniteVolumes.run(model, grid, w₀, cfl=0.2, nb_time_steps=100,
									 numerical_flux=Muscl(limiter=ultrabee))

using Plots
plot(grid, [w₀ w w_muscl w_ultra], 1, label=["initial" "upwind" "minmod" "ultrabee"])