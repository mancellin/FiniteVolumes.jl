#!/usr/bin/env julia
using LinearAlgebra: norm

include("../src/finite_volumes.jl")

grid = PeriodicRegularMesh2D(-1.0, 1.0, 100, -1.0, 1.0, 100)
model = NScalarLinearAdvection{2, Float64, nb_dims(grid)}(@SVector [1.0, 1.0])

function triple_point(i)
	if cell_center(grid, i)[1] < 0.0
		return (1.0, 0.0)
	elseif cell_center(grid, i)[2] < 0.0
		return (0.0, 1.0)
	else
		return (0.0, 0.0)
	end
end

in_rectangle(l, L) = x -> -l/2 <= x[1] <= l/2 && -L/2 <= x[2] <= L/2
in_cross(l, L) = x -> in_rectangle(l, L)(x) || in_rectangle(L, l)(x)
in_circle(r) = x -> norm(x) < r

function jaouen_lagoutiere(i)
	if in_cross(0.4, 1.2)(cell_center(grid, i))
		return (1.0, 0.0)
	elseif in_circle(0.5)(cell_center(grid, i))
		return (0.0, 0.0)
	elseif in_circle(0.7)(cell_center(grid, i))
		return (0.0, 1.0)
	else
		return (0.0, 0.0)
	end
end

#= w₀ = [SVector(triple_point(i)...) for i in 1:nb_cells(grid)] =#
w₀ = [SVector(jaouen_lagoutiere(i)...) for i in 1:nb_cells(grid)]

dt = 0.01
nb_period = 3
nb_time_steps = 2*ceil(Int, nb_period/dt)

#= t, w_upwind = run(model, directional_splitting(grid), w₀, dt=dt, nb_time_steps=nb_time_steps) =#

#= t, w_minmod = run(model, directional_splitting(grid), w₀, dt=dt, nb_time_steps=nb_time_steps, reconstruction=muscl(minmod)) =#

const epsilon = 0.001
mixed_cells(wi) = epsilon <= wi[1] <= (1-epsilon) || epsilon <= wi[2] <= (1-epsilon)

t, w_ultra = run(model, directional_splitting(grid), w₀,
				 dt=dt, nb_time_steps=nb_time_steps,
				 reconstruction=muscl(ultrabee(0.5), mixed_cells))


using PyPlot
colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
function plot_field(grid, w; title=nothing, i_fig=nothing)
	field = Array{Float64}(undef, nb_cells(grid), 3)
	for i in 1:nb_cells(grid)
		field[i, :] = w[i][1]*colors[1] + w[i][2]*colors[2] + (1.0 - w[i][1] - w[i][2])*colors[3]
	end
	field = permutedims(reshape(field, grid.nx, grid.ny, 3), (2, 1, 3))

	if !(i_fig == nothing)
		figure(i_fig)
	else
		figure()
	end
	imshow(field, interpolation="none")
	gca().invert_yaxis()
	if !(title == nothing)
		gca().set_title(title)
	end
end

plot_field(grid, w₀, title="initial", i_fig=1)
plot_field(grid, w_ultra, title="ultrabee", i_fig=2)
#= plot_field(grid, w_upwind, title="upwind") =#
#= plot_field(grid, w_minmod, title="minmod") =#
show()
