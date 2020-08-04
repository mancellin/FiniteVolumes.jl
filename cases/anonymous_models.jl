#!/usr/bin/env julia

using FiniteVolumes
using StaticArrays
using Plots
using Test

# 1 scalar 1D
model = FiniteVolumes.AnonymousModel{1, 1, Float64}(α -> 0.5*α.^2)
grid = RegularMesh1D(0.0, 1.0, 100)
w₀ = [i < nb_cells(grid)/3 ? 0.0 : (i < 2nb_cells(grid)/3 ? 1.0 : 0.0) for i in 1:nb_cells(grid)]
t_final, w = FiniteVolumes.run(model, grid, w₀, cfl=0.2, nb_time_steps=100)
plot(grid, [w₀ w], 1, label=["initial" "upwind"])

# 2 scalars 1D
model = FiniteVolumes.AnonymousModel{2, 1, Float64}(u -> [0.5*u[1].^2 + u[2], u[2]])
grid = RegularMesh1D(0.0, 1.0, 100)
w₀ = SVector{2, Float64}.([i < nb_cells(grid)/3 ? [0.0, 0.0] : (i < 2nb_cells(grid)/3 ? [1.0, 1.0] : [0.0, 0.0]) for i in 1:nb_cells(grid)])
t_final, w = FiniteVolumes.run(model, grid, w₀, cfl=0.2, nb_time_steps=100)
plot(grid, [w₀ w], 1, label=["initial" "upwind"])

# 1 scalars 2D
model = FiniteVolumes.AnonymousModel{1, 2, Float64}(α -> α .* [α, 1.0])
grid = PeriodicRegularMesh2D(50, 50)
in_circle(x) = (x[1]-0.5)^2 + (x[2]-0.5)^2 < 0.3^2
w₀ = [in_circle(FiniteVolumes.cell_center(grid, i)) ? 1.0 : 0.0 for i in 1:FiniteVolumes.nb_cells(grid)]
t_final, w = FiniteVolumes.run(model, grid, w₀, cfl=0.2, nb_time_steps=100)
plot(
     plot(grid, w₀, 1),
     plot(grid, w, 1),
    )

# 2 scalars 2D
model = FiniteVolumes.AnonymousModel{2, 2, Float64}(u -> [
                  [u[1], u[2]^2],
                  [0.0, u[2]],
])
grid = PeriodicRegularMesh2D(50, 50)
in_circle(x) = (x[1]-0.5)^2 + (x[2]-0.5)^2 < 0.3^2
w₀ = SVector{2, Float64}.([in_circle(FiniteVolumes.cell_center(grid, i)) ? [1.0, 1.0] : [0.0, 0.0] for i in 1:FiniteVolumes.nb_cells(grid)])
t_final, w = FiniteVolumes.run(model, grid, w₀, cfl=0.2, nb_time_steps=100)
plot(
     plot(grid, w₀, 1),
     plot(grid, w, 1),
     plot(grid, w₀, 2),
     plot(grid, w, 2),
     layout=(2, 2)
    )
