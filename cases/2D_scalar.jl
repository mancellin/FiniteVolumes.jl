#!/usr/bin/env julia

using StaticArrays
using LinearAlgebra: norm
using FiniteVolumes

grid = PeriodicRegularMesh2D(50, 50)
model = ScalarLinearAdvection{Float64, nb_dims(grid)}(@SVector [1.0, 0.0])

is_in_disk(i) = norm([0.5, 0.5] - cell_center(grid, i)) < 0.3
is_in_square(i, a=0.5) = 0.5-a/2 <= cell_center(grid, i)[1] <= 0.5+a/2 && 0.5-a/2 <= cell_center(grid, i)[2] <= 0.5+a/2

dt = 0.004
nb_period = 1.0
nb_time_steps = ceil(Int, nb_period/dt)

w₀ = [SVector(is_in_square(i) ? 1.0 : 0.0) for i in 1:nb_cells(grid)]

t, w_upwind = FiniteVolumes.run(model, directional_splitting(grid), w₀,
								dt=dt, nb_time_steps=nb_time_steps)

const epsilon = 1e-5
mixed_cells(wi) = epsilon <= wi[1] <= (1.0-epsilon)

#= t, w_minmod = FiniteVolumes.run(model, directional_splitting(grid), w₀, =#
#= 								dt=dt, nb_time_steps=nb_time_steps, =#
#= 								reconstruction=FiniteVolumes.muscl(FiniteVolumes.minmod, mixed_cells)) =#

#= t, w_ultra = FiniteVolumes.run(model, directional_splitting(grid), w₀, =#
#= 							   dt=dt, nb_time_steps=nb_time_steps, =#
#= 							   reconstruction=FiniteVolumes.muscl(FiniteVolumes.ultrabee(0.2))) =#

β = 0.2
using FiniteVolumes: rotation_matrix, rotate_state, stencil
using VOF_ML: youngs_flux, elvira_flux
function vof_reconstruction(method)
    function reconstruction(grid, model, w, wsupp, i_cell, i_face)
        if mixed_cells(w[i_cell])
            α = (x -> x[1]).(w[stencil(grid, i_cell)])
            α_flux = method(α, β)
            return rotate_state([α_flux], wsupp[i_cell], model, rotation_matrix(grid, i_face))
        else
            return rotate_state(w[i_cell], wsupp[i_cell], model, rotation_matrix(grid, i_face))
        end
    end
end

t, w_youngs = FiniteVolumes.run(model, grid, w₀,
								dt=dt, nb_time_steps=nb_time_steps,
                                reconstruction=vof_reconstruction(youngs_flux))
t, w_elvira = FiniteVolumes.run(model, grid, w₀,
								dt=dt, nb_time_steps=nb_time_steps,
                                reconstruction=vof_reconstruction(elvira_flux))

using Plots
plot(
     plot(grid, w₀, 1, title="exact"),
     plot(grid, w_upwind, 1, title="upwind"),
     plot(grid, w_youngs, 1, title="youngs"),
     plot(grid, w_elvira, 1, title="elvira"),
     #= plot(grid, w_minmod, 1, title="minmod"), =#
     #= plot(grid, w_ultra, 1, title="ultrabee"), =#
    )

