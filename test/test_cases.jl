using Test
using StaticArrays
using FiniteVolumes

const ϵ = 1e-15
maximum_principle(w, w₀) = all(minimum(w₀) - ϵ .≤ w .≤ maximum(w₀) + ϵ)
riemann_problem(mesh, w₁, w₂) = [cell_center(mesh, i)[1] < 0.5 ? w₁ : w₂ for i in 1:nb_cells(mesh)]

@testset "Scalar problems" begin
    @testset "Trivial cases" begin
        mesh = RegularMesh1D(0.0, 1.0, 100)

        # No velocity => no change
        w₀ = rand(nb_cells(mesh))
        model = ScalarLinearAdvection(0.0)
        t, w = FiniteVolumes.run(model, mesh, w₀, dt=0.001, nb_time_steps=3)
        @test all(w .== w₀)

        # Uniform initial condition => no change
        w₀ = ones(nb_cells(mesh))
        model = ScalarLinearAdvection(1.0)
        t, w = FiniteVolumes.run(model, mesh, w₀, dt=0.001, nb_time_steps=3)
        @test all(w .== w₀)
    end

    @testset "First order scheme" begin
        mesh = RegularMesh1D(0.0, 1.0, 100)
        sine_w₀ = [sin(cell_center(mesh, i)[1]) for i in 1:nb_cells(mesh)]
        step_w₀ = riemann_problem(mesh, 1.0, 0.0)
        other_step_w₀ = riemann_problem(mesh, 0.0, 1.0)

        forward_model = ScalarLinearAdvection(1.0)
        backward_model = ScalarLinearAdvection(-1.0)

        for w₀ in [sine_w₀, step_w₀, other_step_w₀], m in [forward_model, backward_model]
            t, w = FiniteVolumes.run(m, mesh, w₀, dt=0.001, nb_time_steps=3)
            @test maximum_principle(w, w₀)
        end

        t, wf = FiniteVolumes.run(forward_model, mesh, step_w₀, dt=0.001, nb_time_steps=3)
        t, wb = FiniteVolumes.run(backward_model, mesh, other_step_w₀, dt=0.001, nb_time_steps=3)
        @test all(wf .== reverse(wb))
    end
end

@testset "Euler problems" begin
    @testset "One-fluid shock tubes" begin
        mesh = RegularMesh1D(0.0, 1.0, 100)
        model = IsothermalTwoFluidEuler{1}(1.0, 1.0, 5.0, 1000.0, 1.0)
        w₀ = riemann_problem(mesh, full_state(model, p=2.0, u=0.0, ξ=1.0), full_state(model, p=1.0, u=0.0, ξ=1.0))
        t, w = FiniteVolumes.run(model, mesh, w₀, cfl=0.8, nb_time_steps=20)
        @test isapprox(w[50][1], 1.41, rtol=1e-2)
        @test isapprox(w[50][2], 0.347, rtol=1e-2)
        @test isapprox(w[50][3], 1.0, rtol=1e-2)

        w₀ = riemann_problem(mesh, full_state(model, p=2.0, u=0.0, ξ=0.0), full_state(model, p=1.0, u=0.0, ξ=0.0))
        t, w = FiniteVolumes.run(model, mesh, w₀, cfl=0.8, nb_time_steps=20)
        @test isapprox(w[50][1], 1.5, rtol=1e-2)
        @test isapprox(w[50][2], 1e-4, rtol=1e-2)
        @test isapprox(w[50][3], 0.0, rtol=1e-2)
    end
end
