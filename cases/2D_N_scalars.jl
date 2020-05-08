#!/usr/bin/env julia

using StaticArrays
using LinearAlgebra: norm
using FiniteVolumes

grid = PeriodicRegularMesh2D(-1.0, 1.0, 50, -1.0, 1.0, 50)
model = directional_splitting(ScalarLinearAdvection(3, [1.0, 1.0]))

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
mixed_cells(mesh, model, w, i_face) = any(epsilon .<= w[FiniteVolumes.upwind_cell(grid, model, w, i_face)[2]] .<= (1.0-epsilon))

renormalize(w) = w/(w[1] + w[2] + w[3])

t, w_upwind = FiniteVolumes.run(model, grid, w₀, dt=dt, nb_time_steps=nb_time_steps)

t, w_minmod = FiniteVolumes.run(model, grid, w₀, dt=dt, nb_time_steps=nb_time_steps,
                                numerical_flux=Either(
                                    mixed_cells,
                                    Muscl(limiter=minmod, renormalize=renormalize),
                                    Upwind()
                                ))

t, w_lagout = FiniteVolumes.run(model, grid, w₀, dt=dt, nb_time_steps=nb_time_steps,
                                numerical_flux=LagoutiereDownwind(β=0.1))

@assert all(sum.(w_lagout) .≈ 1.0)

using Plots; gr()
plot(
    plot(grid, w₀, (1, 2, 3)),
    plot(grid, w_upwind, (1, 2, 3)),
    plot(grid, w_minmod, (1, 2, 3)),
    plot(grid, w_lagout, (1, 2, 3)),
   )

