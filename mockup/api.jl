
using FiniteVolumes

mesh = RegularMesh1D(0.0, 1.0, 100)
model = ScalarLinearAdvection(u=1.0)

step(x) = 0.0 < x[1] < 0.3 ? 1.0 : 0.0
u₀ = step.(mesh)

# Anonymous model
dudt(u, _, t) = - div(u -> u^2/2, mesh)(u)

# Anonymous model (scalar 2D)
u = @SVector [0.1, 0.4]
dαdt(α, _, t) = - div(α -> α*u, mesh2d)(α)

# Anonymous model (scalar 2D depending of x and t)
u(x, t) = exp(-t) * @SVector [-x[2], x[1]]
dudt(α, _, t) = - div((α, x) -> α * u(x, t), mesh2d)(α)

# Anonymous model (system 1D)
F(w) = @SVector [w[1] * w[2], w[1] + w[2]]
dαdt(w, _, t) = - div(F, mesh)(w)

# Anonymous model (system 2D)
A = @SMatrix [4.0 1.0; 0.4 0.9]
dαdt(w, _, t) = - div(w -> A * w, mesh2D)(w)

# Existing model
dudt(u, _, t) = - div(ScalarLinearAdvection([1.0, 2.0]), mesh2d)(u)


for i_step in 1:nb_steps
    Δt = time_step(fixed_courant, mesh, w)
    #= CFL = courant(Δt, mesh, w) =#
    using_conservative_variables!(model, w) do v
        v -= Δt * div(flux(model), mesh, method=MUSCL(FVCF(), Ultrabee(Δt=Δt)))(w)
    end
end

