#!/usr/bin/env julia

using FiniteVolumes

mesh = RegularMesh2D(50, 50)
model = ShallowWater{2}()

v₀((x0, y0)) = FiniteVolumes.compute_v.(Ref(model),
                                        [
                                         let (x, y) = FiniteVolumes.cell_center(mesh, i), σ = 0.05
                                             (
                                              h=1.0 + 0.1exp(-((x-x0)^2+(y-y0)^2)/σ^2),
                                              ux=0.0,
                                              uy=0.0
                                             )
                                         end
                                         for i in all_cells(mesh)
                                        ]
                                       )

using Printf
using Plots
# anim = Animation()
# function plot_callback(i, t, w)
#     title = @sprintf "time = %.2f" t
#     plot(mesh, w, :h; title)
#     frame(anim)
# end
# t, w = FiniteVolumes.run(model, mesh, w₀; cfl=0.4, nb_time_steps=100, callback=plot_callback)
# mp4(anim, "anim.mp4")

using OrdinaryDiffEq
dvdt(v, p, t) = -FiniteVolumes.div(model, mesh, FiniteVolumes.invert_v.(Ref(model), v), numerical_flux=Upwind())
final_time = 0.5

secret_position = (0.4, 0.8)
secret_sol = solve(ODEProblem(dvdt, v₀(secret_position), final_time), Euler(), dt=0.002, saveat=0.01)

# plot(mesh, sol(0.05), 1)

function measure(sol) 
    height_in_cell(i) = (v->v[1]).(sol[i, :])
    [height_in_cell(42) height_in_cell(1001) height_in_cell(1989)]
end
measured_heights = measure(secret_sol)
plot(secret_sol.t, measure(secret_sol), marker=:circle, label=[42 1001 1989])

function loss(xy)
    sol = solve(ODEProblem(dvdt, v₀(xy), final_time), Euler(), dt=0.002, saveat=0.01)
    sum((measure(sol) .- measured_heights).^2)
end

# using ForwardDiff
# ForwardDiff.gradient(loss, [0.5, 0.5])

using Optim
cb(state) = (println(state); false)
lossd = OnceDifferentiable(loss, [0.5, 0.5]; autodiff = :forward);
opt = optimize(lossd, [0.5, 0.5], BFGS(), Optim.Options(store_trace=false, callback=cb))
