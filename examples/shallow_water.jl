using FiniteVolumes, StaticArrays, OrdinaryDiffEq, Plots

mesh = PeriodicCartesianMesh(50, 50)
u0 = [SVector(1.0 + exp(-100*((x - 0.4)^2 + (y - 0.3)^2)), 0.0, 0.0) for (x, y) in cell_centers(mesh)]

function f(v, n)
    h, ux, uy = v[1], v[2]/v[1], v[3]/v[1]
    un = ux*n[1] + uy*n[2]
    return SVector(h*un, h*ux*un + h^2*n[1]*9.81/2, h*uy*un + h^2*n[2]*9.81/2)
end

dudt(u, t, p) = -FiniteVolumes.div(f, mesh, u, Upwind())

problem = ODEProblem(dudt, u0, (0.0, 0.2))
sol = solve(problem, Euler(), dt=0.001, saveat=0.01)

plot(
    plot(mesh, sol(0.0), 1),
    plot(mesh, sol(0.06), 1),
    plot(mesh, sol(0.13), 1),
    plot(mesh, sol(0.20), 1),
)
