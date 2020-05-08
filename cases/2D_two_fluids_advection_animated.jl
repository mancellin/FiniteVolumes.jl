using LinearAlgebra: norm
using ProgressMeter: @showprogress
using FiniteVolumes
using Plots

function main()
    grid = PeriodicRegularMesh2D(50, 50)
    model = IsothermalTwoFluidEuler{nb_dims(grid)}(300.0, 1.0, 1500.0, 1000.0, 1e5)

    is_in_disk(i) = norm([0.5, 0.5] - cell_center(grid, i)) < 0.3

    w₀ = [full_state(model, p=1e5, ux=10.0, uy=10.0, ξ=(is_in_disk(i) ? 0.0 : 1.0))
          for i in 1:nb_cells(grid)]

    w = deepcopy(w₀)

    t = 0.0
    dt = 3e-6
    anim = @animate for x = 1:1_000
        FiniteVolumes.using_conservative_variables!(model, w) do v
            v .-= dt * FiniteVolumes.div(model, grid, w)
        end
        t += dt
        plot(grid, w, :p, clim=(1e5-1e-2, 1e5+1e-2))
    end every 10
    gif(anim, "test.gif")
end

main()
