#!/usr/bin/env julia

using StaticArrays
using LinearAlgebra: norm
using FiniteVolumes

grid = PeriodicRegularMesh2D(50, 50)
model = ScalarLinearAdvection{Float64, nb_dims(grid)}(@SVector [1.0, 1.0])

is_in_disk(i) = norm([0.5, 0.5] - cell_center(grid, i)) < 0.3

dt = 0.004
nb_period = 1.0
nb_time_steps = ceil(Int, nb_period/dt)

w₀ = [SVector(is_in_disk(i) ? 1.0 : 0.0) for i in 1:nb_cells(grid)]

t, w_upwind = FiniteVolumes.run(model, directional_splitting(grid), w₀,
								dt=dt, nb_time_steps=nb_time_steps)

const epsilon = 1e-5
mixed_cells(wi) = epsilon <= wi[1] <= (1.0-epsilon)

t, w_minmod = FiniteVolumes.run(model, directional_splitting(grid), w₀,
								dt=dt, nb_time_steps=nb_time_steps,
								reconstruction=FiniteVolumes.muscl(FiniteVolumes.minmod, mixed_cells))

t, w_ultra = FiniteVolumes.run(model, directional_splitting(grid), w₀,
							   dt=dt, nb_time_steps=nb_time_steps,
							   reconstruction=FiniteVolumes.muscl(FiniteVolumes.ultrabee(0.2)))


using PyPlot
function plot_field(grid, w; title=nothing, i_fig=nothing)
    field = transpose(reshape([wi[1] for wi in w], (grid.nx, grid.ny)))

	if !(i_fig == nothing)
		figure(i_fig)
	else
		figure()
	end

	imshow(field, interpolation="none", vmin=0.0, vmax=1.0)

	if !(title == nothing)
		gca().set_title(title)
	end
    #= colorbar() =#
end

plot_field(grid, w₀, title="initial", i_fig=1)
plot_field(grid, w_upwind, title="upwind", i_fig=2)
plot_field(grid, w_minmod, title="minmod", i_fig=3)
plot_field(grid, w_ultra, title="ultrabee", i_fig=4)
show()

