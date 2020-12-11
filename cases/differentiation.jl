#!/usr/bin/env julia

using FiniteVolumes

grid = PeriodicRegularMesh2D(20, 20)
model = directional_splitting(ScalarLinearAdvection([1.0, 1.0]))

w₀(w) = [((x, y) = FiniteVolumes.cell_center(grid, i); w*sin(2π*x)*cos(2π*y)) for i in 1:nb_cells(grid)]

limiter(p) = (a, b, β) -> p*FiniteVolumes.minmod(a, b, β)
numerical_flux(p) = Muscl(limiter=limiter(p))

# using Plots
# tf, wf = FiniteVolumes.run(model, grid, w₀(1.0), cfl=0.2, nb_time_steps=100, numerical_flux=numerical_flux(0.0))
# tf, w2 = FiniteVolumes.run(model, grid, w₀(1.0), cfl=0.2, nb_time_steps=100, numerical_flux=numerical_flux(1.0))
# plot(plot(grid, wf, 1), plot(grid, w2, 1)) |> display

using ForwardDiff

function loss(p)
    w0 = w₀(p[1])
    wf = FiniteVolumes.run(model, grid, w0, cfl=0.2, nb_time_steps=100, numerical_flux=numerical_flux(p[2]))[2]
    sum((wf .- w0).^2)
end
@show loss([1.0, 0.0])
@show loss([1.0, 1.0])
function train()
    p = [1.0, 0.0]
    for i in 1:2
        @show g = ForwardDiff.gradient(loss, p)[2]
        p[2] = p[2] - 0.001 * g
    end
    return p
end
train()
