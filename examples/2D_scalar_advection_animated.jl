using FiniteVolumes
using Printf
using Plots

mesh = PeriodicCartesianMesh(40, 40)
model = LinearAdvectionFlux([1.0, 1.0])

is_in_square(x, side=0.5) = all(0.5-side/2 .<= x .<= 0.5+side/2)

w₀ = [is_in_square(x) ? 1.0 : 0.0 for x in cell_centers(mesh)]

anim = Animation()
function plot_callback(i, t, w)
    if i % 2 == 0
        title = @sprintf "time = %.2f" t
        plot(mesh, w, 1, title=title, clims=(0, 1))
        frame(anim)
    end
end

t, w = FiniteVolumes.run(directional_splitting(model), mesh, w₀,
                         time_step=FixedCourant(0.2),
                         nb_time_steps=200,
                         callback=plot_callback)
gif(anim, "advection.gif")
