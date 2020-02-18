#!/usr/bin/env julia

using StaticArrays
using FiniteVolumes

grid = RegularMesh1D(0.0, 1.0, 100)
model = ScalarLinearAdvection(1.0)

w₀ = [SVector(i < nb_cells(grid)/2 ? 1.0 : 0.0) for i in 1:nb_cells(grid)]

t_final, w = FiniteVolumes.run(model, grid, w₀, cfl=0.2, nb_time_steps=100)

t_final, w_muscl = FiniteVolumes.run(model, grid, w₀, cfl=0.2, nb_time_steps=100,
									 reconstruction=FiniteVolumes.muscl(FiniteVolumes.minmod))

t_final, w_ultra = FiniteVolumes.run(model, grid, w₀, cfl=0.2, nb_time_steps=100,
									 reconstruction=FiniteVolumes.muscl(FiniteVolumes.ultrabee(0.2)))

using PyPlot
figure()
x = [cell_center(grid, i) for i in 1:nb_cells(grid)]
step(x, [wi[1] for wi in w₀], label="Initial")
step(x, [wi[1] for wi in w], label="Upwind")
step(x, [wi[1] for wi in w_muscl], label="Minmod")
step(x, [wi[1] for wi in w_ultra], label="Ultrabee")
legend()
show()
