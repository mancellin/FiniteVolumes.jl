#!/usr/bin/env julia
"""Diagonal advection of a disk of liquid into gas"""

include("../src/finite_volumes.jl")

grid = PeriodicRegularMesh2D(50, 50)
model = IsothermalTwoFluidEuler{Float64, nb_dims(grid)}(300.0, 1.0, 1500.0, 1000.0, 1e5)

using LinearAlgebra: norm
is_in_disk(i) = norm([0.5, 0.5] - cell_center(grid, i)) < 0.3

w = [SVector(1e5, 10.0, 10.0, is_in_disk(i) ? 0.0 : 1.0) for i in 1:nb_cells(grid)]
wsupp = map(wi -> compute_wsupp(model, wi), w)
run!(model, directional_splitting(grid), w, wsupp, dt=1e-6, nb_time_steps=1_000)
#= run!(model, grid, w, wsupp, dt=1e-6, nb_time_steps=1_000) =#

using PyPlot: figure, plot, imshow, colorbar
function plot_field(grid, w, i)
    field = transpose(reshape([wi[i] for wi in w], (grid.nx, grid.ny)))
    imshow(field) #, vmin=0.0, vmax=1.0)
    colorbar()
end
figure()
plot_field(grid, w, 1)
figure()
plot_field(grid, wsupp, 4)
show()

