#!/usr/bin/env julia

using FiniteVolumes

grid = RegularMesh1D(0.0, 1.0, 100)
model = IsothermalTwoFluidEuler{nb_dims(grid)}(300.0, 1.0, 1500.0, 1000.0, 1e5)

left_state = full_state(model, p=2e5, u=0.0, ξ=1.0)
right_state = full_state(model, p=1e5, u=0.0, ξ=0.0)
w₀ = [i < nb_cells(grid)/2 ? left_state : right_state for i in all_cells(grid)]

t, w = FiniteVolumes.run(model, grid, w₀, cfl=0.4, nb_time_steps=100)

using Plots
plot(grid, [w₀ w], :p, label=["initial" "final"])
