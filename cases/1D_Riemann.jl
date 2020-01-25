#!/usr/bin/env julia

include("../src/finite_volumes.jl")

grid = RegularMesh1D(0.0, 1.0, 100)
model = IsothermalTwoFluidEuler{Float64, nb_dims(grid)}(300.0, 1.0, 1500.0, 1000.0, 1e5)

w₀ = [i < nb_cells(grid)/2 ? SVector(2e5, 0.0, 1.0) : SVector(1e5, 0.0, 0.0) for i in 1:nb_cells(grid)]

w = deepcopy(w₀)
wsupp = map(wi -> compute_wsupp(model, wi), w)
run!(model, grid, w, wsupp, cfl=0.4, nb_time_steps=100)

using PyPlot
figure()
step([cell_center(grid, i) for i in 1:nb_cells(grid)], [wi[1] for wi in w₀])
step([cell_center(grid, i) for i in 1:nb_cells(grid)], [wi[1] for wi in w])
show()
