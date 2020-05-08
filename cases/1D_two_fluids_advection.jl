#!/usr/bin/env julia

using FiniteVolumes

grid = RegularMesh1D(0.0, 1.0, 100)
model = IsothermalTwoFluidEuler{Float64, nb_dims(grid)}(300.0, 1.0, 1500.0, 1000.0, 1e5)

w₀ = [full_state(model, p=1e5, u=10.0, ξ=
              0.2 < cell_center(grid, i)[1] < 0.5 ? 0.0 : 1.0
             ) for i in 1:nb_cells(grid)]

t_final, w = FiniteVolumes.run(model, grid, w₀, cfl=0.4, nb_time_steps=10_000)

using Plots
plot(
     plot(grid, [w₀ w], :p, label=["initial" "final"]),
     plot(grid, [w₀ w], :u, label=["initial" "final"]),
     plot(grid, [w₀ w], :ξ, label=["initial" "final"]),
     layout=(3, 1)
    )
