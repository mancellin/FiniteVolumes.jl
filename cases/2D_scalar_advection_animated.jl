using LinearAlgebra: norm
using ProgressMeter: @showprogress
using FiniteVolumes
using Printf
using Plots

# mesh = PeriodicRegularMesh2D(40, 40)
# model = directional_splitting(ScalarLinearAdvection([1.0, 1.0]))

mesh = RegularMesh2D(40, 40)
u(x, center=(0.5, 0.5)) = [-(x[2]-center[2]), (x[1]-center[1])]
model = FiniteVolumes.AnonymousModel{1, 2, Float64, true}((α, x) -> α .* u(x)) 

is_in_square(i, side=0.5) = all(0.5-side/2 .<= cell_center(mesh, i) .<= 0.5+side/2)

w₀ = [is_in_square(i) ? 1.0 : 0.0 for i in 1:nb_cells(mesh)]

anim = Animation()
function plot_callback(i, t, w)
    if i % 2 == 0
        title = @sprintf "time = %.2f" t
        plot(mesh, w, 1, title=title, clims=(0, 1))
        frame(anim)
    end
end

t, w = FiniteVolumes.run(model, mesh, w₀, cfl=0.2, nb_time_steps=100, callback=plot_callback)

gif(anim, "advection.gif")
