#!/usr/bin/env julia

using FiniteVolumes
using DifferentialEquations

mesh = RegularMesh1D(0.0, 1.0, 100)
model = ScalarLinearAdvection(1.0)

w₀ = [i < nb_cells(mesh)/4 ? 1.0 : 0.0 for i in 1:nb_cells(mesh)]
# w₀ = [sin(2π*cell_center(mesh, i)[1]) for i in 1:nb_cells(mesh)]

dwdt_upwind(w, p, t) = -FiniteVolumes.div(model, mesh, numerical_flux=Upwind())(w)
prob = ODEProblem(dwdt_upwind, w₀, 0.4)
sol1 = solve(prob, Euler(), dt=0.005, saveat=0.1)
sol2 = solve(prob, RK4(), dt=0.005, adaptive=false, saveat=0.1)
sol3 = solve(prob, ImplicitEuler(), dt=0.1, saveat=0.1)

dwdt_muscl(w, p, t) = -FiniteVolumes.div(model, mesh, numerical_flux=Muscl(limiter=minmod))(w)
prob = ODEProblem(dwdt_muscl, w₀, 0.4)
sol4 = solve(prob, Euler(), dt=0.005, saveat=0.1)
sol5 = solve(prob, RK4(), dt=0.005, adaptive=false, saveat=0.1)
sol6 = solve(prob, ImplicitEuler(), dt=0.1, saveat=0.1)

using Plots
plot(mesh, sol1(0.0), 1, color=:grey, linestyle=:dash, label="Initial")
plot!(mesh, [sol1(0.2) sol1(0.4)], 1, color=:red,    label="Upwind+Explicit")
plot!(mesh, [sol2(0.2) sol2(0.4)], 1, color=:green,  label="Upwind+RK4")
plot!(mesh, [sol3(0.2) sol3(0.4)], 1, color=:purple, label="Upwind+Implicit")
plot!(mesh, [sol4(0.2) sol4(0.4)], 1, color=:orange, label="Minmod+Explicit")
plot!(mesh, [sol5(0.2) sol5(0.4)], 1, color=:blue,   label="Minmod+RK4")
plot!(mesh, [sol6(0.2) sol6(0.4)], 1, color=:pink,   label="Minmod+Implicit")

