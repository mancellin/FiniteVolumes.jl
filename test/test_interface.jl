
using Test
using FiniteVolumes

@testset "Interface with other packages" begin
    @testset "Measurements" begin
        using Measurements
        mesh = CartesianMesh(10)
        w0 = map(x -> sin(2π*x), cell_centers(mesh))
        i_face = (FiniteVolumes.CartesianMeshes.Half(11),)

        flux = LinearAdvectionFlux(1.0 ± 1.0)
        @test (Upwind())(flux, mesh, w0, i_face) |> Measurements.value == w0[5]

        dt = 0.01
        w = w0 .- dt*FiniteVolumes.div(flux, mesh, w0)
    end

    @testset "Unitful" begin
        using Unitful: kg, m, s
        mesh = CartesianMesh(0.0m, 1.0m, 10)
        i_face = (FiniteVolumes.CartesianMeshes.Half(11),)
        dt = 0.01s

        flux = LinearAdvectionFlux(1m/s)
        w1 = map(x -> sin(2π*x/1m)*1kg/m, cell_centers(mesh))
        @test (Upwind())(flux, mesh, w1, i_face) |> typeof == typeof(w1[1]* 1m/s)
        w2 = w1 .- dt*FiniteVolumes.div(flux, mesh, w1)
        @test eltype(w2) == eltype(w1)

    #     flux = Wave1DFlux(1m/s)
    #     w1 = map(x -> (sin(2π*x/1m)*kg/m, 0kg/m/s), cell_centers(mesh))
    #     Δw = map(x -> (0kg/m/s, 0kg/m/s^2), cell_centers(mesh))
    #     w2 = w1 .- dt.*FiniteVolumes.div(flux, mesh, w1)
    #     @test eltype(w2) == eltype(w1)
    end

    @testset "ForwardDiff" begin
        using ForwardDiff
        function run(v)
            f = LinearAdvectionFlux(v)
            mesh = CartesianMesh(0.0, 1.0, 100)
            w = [sin(2π*x) for x in cell_centers(mesh)]
            dt = 0.01
            w2 = w .- dt*FiniteVolumes.div(f, mesh, w)
        end
        ForwardDiff.derivative(run, 1.0)
    end

    @testset "OrdinaryDiffEq" begin
        using OrdinaryDiffEq
        mesh = CartesianMesh(10)
        flux = LinearAdvectionFlux(1.0)

        w₀ = map(x -> x < 0.5 ? 1.0 : 0.0, cell_centers(mesh))
        dwdt_upwind(w, p, t) = -FiniteVolumes.div(flux, mesh, w)
        prob = ODEProblem(dwdt_upwind, w₀, 0.4)
        sol1 = solve(prob, Euler(), dt=0.005, saveat=0.1)
        sol2 = solve(prob, RK4(), dt=0.005, adaptive=false, saveat=0.1)
        sol3 = solve(prob, ImplicitEuler(), dt=0.1, saveat=0.1)

        @test all(0.0 .<= sol1.u[end] .<= 1.0)
        @test all(0.0 .<= sol2.u[end] .<= 1.0)
        @test all(0.0 .<= sol2.u[end] .<= 1.0)
    end

    @testset "Implicit heat equation" begin
        using OrdinaryDiffEq
        mesh = PeriodicCartesianMesh(10)
        w₀ = map(x -> exp(-1000*(x - 0.5)^2), cell_centers(mesh))
        dwdt(w, p, t) = -FiniteVolumes.div(FiniteVolumes.∇, mesh, w, FiniteVolumes.Centered())
        prob = ODEProblem(dwdt, w₀, (0, 0.1))
        sol = solve(prob, ImplicitEuler(), dt=0.02, saveat=0.02)
    end
end
