#!/usr/bin/env julia

using StaticArrays
using LinearAlgebra: norm
using FiniteVolumes

grid = PeriodicRegularMesh2D(-1.0, 1.0, 50, -1.0, 1.0, 50)
model = NScalarLinearAdvection{3, Float64, nb_dims(grid)}(@SVector [1.0, 0.5])

function triple_point(i)
	if cell_center(grid, i)[1] < 0.0
		return (1.0, 0.0, 0.0)
	elseif cell_center(grid, i)[2] < 0.0
		return (0.0, 1.0, 0.0)
	else
		return (0.0, 0.0, 1.0)
	end
end

in_rectangle(l, L) = x -> -l/2 <= x[1] <= l/2 && -L/2 <= x[2] <= L/2
in_cross(l, L) = x -> in_rectangle(l, L)(x) || in_rectangle(L, l)(x)
in_circle(r) = x -> norm(x) < r

function jaouen_lagoutiere(i)
	if in_cross(0.4, 1.2)(cell_center(grid, i))
		return (0.0, 1.0, 0.0)
	elseif in_circle(0.5)(cell_center(grid, i))
		return (0.0, 0.0, 1.0)
	elseif in_circle(0.7)(cell_center(grid, i))
		return (1.0, 0.0, 0.0)
	else
		return (0.0, 0.0, 1.0)
	end
end

#= w₀ = [SVector(triple_point(i)...) for i in 1:nb_cells(grid)] =#
w₀ = [SVector(jaouen_lagoutiere(i)...) for i in 1:nb_cells(grid)]

dt = 0.004
nb_period = 2
nb_time_steps = 2*ceil(Int, nb_period/dt)

const epsilon = 1e-5
mixed_cells(wi) = epsilon <= wi[1] <= (1.0-epsilon) || epsilon <= wi[2] <= (1.0-epsilon) || epsilon <= wi[3] <= (1.0-epsilon)

t, w_upwind = FiniteVolumes.run(model, directional_splitting(grid), w₀,
								dt=dt, nb_time_steps=nb_time_steps)

renormalize(w) = w/(w[1] + w[2] + w[3])

t, w_minmod = FiniteVolumes.run(model, directional_splitting(grid), w₀,
								dt=dt, nb_time_steps=nb_time_steps,
								reconstruction=FiniteVolumes.muscl(FiniteVolumes.minmod, mixed_cells, renormalize))

t, w_ultra = FiniteVolumes.run(model, directional_splitting(grid), w₀,
							   dt=dt, nb_time_steps=nb_time_steps,
							   reconstruction=FiniteVolumes.muscl(FiniteVolumes.ultrabee(0.1), mixed_cells, renormalize))


using PyPlot
colors = [[0.3, 1.0, 0.0], [0.3, 0.0, 1.0], [0.0, 0.0, 0.0]]
function plot_field(grid, w; title=nothing, i_fig=nothing)
	field = Array{Float64}(undef, nb_cells(grid), 3)
	for i in 1:nb_cells(grid)
		if !(w[i][1] + w[i][2] + w[i][3] ≈ 1.0)
			field[i, :] = [1.0, 0.0, 0.0] # red
		else
			field[i, :] = w[i][1]*colors[1] + w[i][2]*colors[2] + (1.0 - w[i][1] - w[i][2])*colors[3]
		end
	end
	field = permutedims(reshape(field, grid.nx, grid.ny, 3), (2, 1, 3))

	if !(i_fig == nothing)
		figure(i_fig)
	else
		figure()
	end
	imshow(field, interpolation="none")
	if !(title == nothing)
		gca().set_title(title)
	end
end

plot_field(grid, w₀, title="initial", i_fig=1)
plot_field(grid, w_upwind, title="upwind", i_fig=2)
plot_field(grid, w_minmod, title="minmod", i_fig=3)
plot_field(grid, w_ultra, title="ultrabee", i_fig=4)
show()
