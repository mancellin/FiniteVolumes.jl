using StaticArrays
using LinearAlgebra: norm
using ProgressMeter: @showprogress
using FiniteVolumes
using Plots

grid = PeriodicRegularMesh2D(50, 50)
model = IsothermalTwoFluidEuler{Float64, nb_dims(grid)}(300.0, 1.0, 1500.0, 1000.0, 1e5)

is_in_disk(i) = norm([0.5, 0.5] - cell_center(grid, i)) < 0.3

w₀ = [SVector(1e5, 10.0, 10.0, is_in_disk(i) ? 0.0 : 1.0) for i in 1:nb_cells(grid)]

w = deepcopy(w₀)
wsupp = map(wi -> FiniteVolumes.compute_wsupp(model, wi), w)

t = 0.0
anim = @animate for x = 1:1_000
    (dt, cfl) = FiniteVolumes.update!(model, directional_splitting(grid), w, wsupp; dt=3e-6)
    global t += dt
    plot(grid, w, 1, clim=(1e5-1e-2, 1e5+1e-2))
end every 10
gif(anim, "test.gif")

