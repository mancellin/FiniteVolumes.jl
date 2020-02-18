
using FiniteVolumes

mesh = RegularMesh1D(0.0, 1.0, 100)
model = ScalarLinearAdvection(u=1.0)

step(x) = 0.0 < x[1] < 0.3 ? 1.0 : 0.0
u₀ = step.(mesh)

dudt(u, F, t) = - F(u)

pb = FVProblem(dudt, u₀, model, NeumannBC(), tspan=(0.0, 1.0))
res = solve(pb, Scheme(RK2(), Upwind(), reconstruction=MUSCL(Minmod())), dt=0.001)

using Plots

plot(res[end])
