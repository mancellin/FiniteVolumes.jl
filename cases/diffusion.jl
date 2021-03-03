using OrdinaryDiffEq
using FiniteVolumes
using Plots

struct DiffusionFlux <: FiniteVolumes.AbstractFlux end

function (::FiniteVolumes.Centered)(::DiffusionFlux, mesh, w, i_face)
    n = FiniteVolumes.normal_vector(mesh, i_face)
    Δx = n[1] > n[2] ? FiniteVolumes.dx(mesh)[1] : FiniteVolumes.dx(mesh)[2]
    i_cell_1, i_cell_2 = FiniteVolumes.cells_next_to_inner_face(mesh, i_face)
    return (w[i_cell_1] - w[i_cell_2])/Δx
end

mesh = PeriodicCartesianMesh(20, 20)
w₀ = map(x -> 100*exp(-1000*((x[1]-0.5)^2 + (x[2]-0.5)^2)), cell_centers(mesh))

dwdt(w, p, t) = -FiniteVolumes.div(DiffusionFlux(), mesh, w, FiniteVolumes.Centered())
prob = ODEProblem(dwdt, w₀, (0, 0.1))
sol = solve(prob, ImplicitEuler(), dt=0.01, saveat=0.01)

plot(
     plot(mesh, sol(0.0), 1, clims=(0, 1)),
     plot(mesh, sol(0.033), 1, clims=(0, 1)),
     plot(mesh, sol(0.066), 1, clims=(0, 1)),
     plot(mesh, sol(0.1), 1, clims=(0, 1)),
    )
