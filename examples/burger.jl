using FiniteVolumes, OrdinaryDiffEq, Plots

mesh = PeriodicCartesianMesh(100)
u0 = [sin(2π*x) for x in cell_centers(mesh)]
F(u) = u^2 / 2

dudt(u, t, p) = -FiniteVolumes.div(F, mesh, u, Upwind())
problem = ODEProblem(dudt, u0, (0.0, 0.5))
sol = solve(problem, Euler(), dt=1e-3, saveat=0.05)

plot(mesh, [u0 sol(0.25) sol(0.5)],
     label=["initial condition" "middle state" "final state"])
