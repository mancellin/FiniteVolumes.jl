
using StaticArrays
using FiniteVolumes
using DifferentialEquations

mesh = RegularMesh1D(0.0, 1.0, 100)
model = ScalarLinearAdvection(1.0)

#= w₀ = [SVector{1, Float64}(i < nb_cells(mesh)/3 ? 1.0 : 0.0) for i in 1:nb_cells(mesh)] =#
w₀ = [SVector{1, Float64}(sin(2π*cell_center(mesh, i)[1])) for i in 1:nb_cells(mesh)]
wsupp = map(wi -> FiniteVolumes.compute_wsupp(model, wi), w₀)

function dwdt(w, p, t)
    dw, λmax = FiniteVolumes.balance(model, mesh, w, wsupp; reconstruction=p)
    return dw
end

prob = ODEProblem(dwdt, w₀, 0.4, FiniteVolumes.no_reconstruction)
sol1 = solve(prob, Euler(), dt=0.005, saveat=0.1)
sol2 = solve(prob, RK4(), dt=0.005, adaptive=false, saveat=0.1)

prob = ODEProblem(dwdt, w₀, 0.4, FiniteVolumes.muscl(FiniteVolumes.minmod))
sol3 = solve(prob, Euler(), dt=0.005, saveat=0.1)
sol4 = solve(prob, RK4(), dt=0.005, adaptive=false, saveat=0.1)

using PyPlot
extract_profile(sol_at_t) = (w -> w[1]).(sol_at_t)
plot(extract_profile(sol1(0.0)))
plot(extract_profile(sol1(0.2)), label="Euler")
plot(extract_profile(sol1(0.4)), label="Euler")
plot(extract_profile(sol2(0.2)), label="RK4")
plot(extract_profile(sol2(0.4)), label="RK4")
plot(extract_profile(sol3(0.2)), label="Minmod")
plot(extract_profile(sol3(0.4)), label="Minmod")
plot(extract_profile(sol4(0.2)), label="Minmod+RK4")
plot(extract_profile(sol4(0.4)), label="Minmod+RK4")
legend()
