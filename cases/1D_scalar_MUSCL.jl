#!/usr/bin/env julia

include("../src/finite_volumes.jl")

grid = RegularMesh1D(0.0, 1.0, 100)
model = ScalarLinearAdvection(-1.0)

const cfl = 0.378

minmod(a, b) = a*b <= 0 ? 0.0 : (a >= 0 ? min(a, b) : max(a, b))
superbee(a, b) = a*b <= 0 ? 0.0 : (a >= 0 ? max(min(2*a, b), min(a, 2*b)) : min(max(2*a, b), max(a, 2*b)))
ultrabee(a, b) = a*b <= 0 ? 0.0 : (a >= 0 ? 2*max(0, min((1/cfl-1)*a, b)) : -ultrabee(-a, -b))

limiter = minmod

function state_in_cell_at_face(grid::RegularMesh1D, w, wsupp, model, i_cell, i_face)
    left_∇w = left_gradient(grid, w, i_cell)
    right_∇w = right_gradient(grid, w, i_cell)
    if face_center(grid, i_face)[1] > cell_center(grid, i_cell)[1]
        limited_∇w = limiter.(left_∇w, right_∇w)
        reconstructed_w = w[i_cell] + dx(grid)/2*limited_∇w
    else
        limited_∇w = limiter.(right_∇w, left_∇w)
        reconstructed_w = w[i_cell] - dx(grid)/2*limited_∇w
    end
    reconstructed_wsupp = compute_wsupp(model, reconstructed_w)
    return rotate_state(reconstructed_w, reconstructed_wsupp, model, rotation_matrix(grid, i_face))
end

w₀ = [SVector(i < nb_cells(grid)/2 ? 1.0 : 0.0) for i in 1:nb_cells(grid)]

w = deepcopy(w₀)
wsupp = map(wi -> compute_wsupp(model, wi), w)
run!(model, grid, w, wsupp, cfl=cfl, nb_time_steps=100)

using PyPlot
figure()
step([cell_center(grid, i) for i in 1:nb_cells(grid)], [wi[1] for wi in w₀])
step([cell_center(grid, i) for i in 1:nb_cells(grid)], [wi[1] for wi in w])
show()
