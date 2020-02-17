#!/usr/bin/env julia
using LinearAlgebra: norm

include("../src/finite_volumes.jl")

grid = PeriodicRegularMesh2D(50, 50)
model = NScalarLinearAdvection{2, Float64, nb_dims(grid)}(@SVector [1.0, 0.5])

function triple_point(i)
	if cell_center(grid, i)[1] < 0.5
		return (1.0, 0.0)
	elseif cell_center(grid, i)[2] < 0.4
		return (0.0, 1.0)
	else
		return (0.0, 0.0)
	end
end

w₀ = [SVector(triple_point(i)...) for i in 1:nb_cells(grid)]

w = deepcopy(w₀)
wsupp = map(wi -> compute_wsupp(model, wi), w)
run!(model, grid, w, wsupp, cfl=0.2, nb_time_steps=10)

using PyPlot
function plot_field(grid, w)
	field = Array{Float64}(undef, grid.nx*grid.ny, 3)
	for i in 1:2500
		field[i, :] = [w[i][1], w[i][2], 1.0 - w[i][1] - w[i][2]]
	end
	field = permutedims(reshape(field, grid.nx, grid.ny, 3), (2, 1, 3))
    imshow(field, interpolation="none")
end
figure()
plot_field(grid, w)
show()
