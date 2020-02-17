#!/usr/bin/env julia

include("../src/finite_volumes.jl")

grid = RegularMesh1D(0.0, 1.0, 100)
model = ScalarLinearAdvection(-1.0)

w₀ = [SVector(i < nb_cells(grid)/2 ? 1.0 : 0.0) for i in 1:nb_cells(grid)]

w = deepcopy(w₀)
wsupp = map(wi -> compute_wsupp(model, wi), w)
run!(model, grid, w, wsupp, cfl=cfl, nb_time_steps=100)

using PyPlot
figure()
step([cell_center(grid, i) for i in 1:nb_cells(grid)], [wi[1] for wi in w₀])
step([cell_center(grid, i) for i in 1:nb_cells(grid)], [wi[1] for wi in w])
show()
