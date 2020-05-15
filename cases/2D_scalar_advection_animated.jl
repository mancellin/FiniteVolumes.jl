using LinearAlgebra: norm
using ProgressMeter: @showprogress
using FiniteVolumes
using Printf
using Plots

mesh = PeriodicRegularMesh2D(40, 40)
model = directional_splitting(ScalarLinearAdvection([1.0, 1.0]))

is_in_square(i, a=0.5) = all(0.5-a/2 .<= cell_center(mesh, i) .<= 0.5+a/2)

w₀ = [is_in_square(i) ? 1.0 : 0.0 for i in 1:nb_cells(mesh)]

anim = Animation()
function plot_callback(i, t, w)
    if i % 2 == 0
        title = @sprintf "time = %.2f" t
        plot(mesh, w, 1, title=title, clims=(0, 1))
        frame(anim)
    end
end

FiniteVolumes.run(model, mesh, w₀, cfl=0.2, nb_time_steps=200, callback=plot_callback)

gif(anim, "advection.gif")
