#!/usr/bin/env julia

using StaticArrays
using FiniteVolumes
using FiniteVolumes: div, first_order_upwind, muscl, minmod
using DifferentialEquations

mesh = RegularMesh1D(0.0, 1.0, 100)
model = ScalarLinearAdvection(1.0)

w₀ = [SVector{1, Float64}(i < nb_cells(mesh)/4 ? 1.0 : 0.0) for i in 1:nb_cells(mesh)]
#= w₀ = [SVector{1, Float64}(sin(2π*cell_center(mesh, i)[1])) for i in 1:nb_cells(mesh)] =#

dwdt_upwind(w, p, t) = -div(model, mesh, numerical_flux=first_order_upwind)(w)
prob = ODEProblem(dwdt_upwind, w₀, 0.4)
sol1 = solve(prob, Euler(), dt=0.005, saveat=0.1)
sol2 = solve(prob, RK4(), dt=0.005, adaptive=false, saveat=0.1)

dwdt_muscl(w, p, t) = -div(model, mesh, numerical_flux=muscl(minmod))(w)
prob = ODEProblem(dwdt_muscl, w₀, 0.4)
sol3 = solve(prob, Euler(), dt=0.005, saveat=0.1)
sol4 = solve(prob, RK4(), dt=0.005, adaptive=false, saveat=0.1)

using Plots
plot(mesh, sol1(0.0), 1, color=:grey, linestyle=:dash, label="Initial")
plot!(mesh, [sol1(0.2) sol1(0.4)], 1, color=:blue, label="Upwind+Euler")
plot!(mesh, [sol2(0.2) sol2(0.4)], 1, color=:red, label="Upwind+RK4")
plot!(mesh, [sol3(0.2) sol3(0.4)], 1, color=:green, label="Minmod+Euler")
plot!(mesh, [sol4(0.2) sol4(0.4)], 1, color=:purple, label="Minmod+RK4")

