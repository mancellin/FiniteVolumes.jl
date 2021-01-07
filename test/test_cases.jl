# Integration tests on some simple test cases.

using Test
using LinearAlgebra: norm
using StaticArrays
using FiniteVolumes

# TOOLS
getfield(w, var) = [wi[var] for wi in w]

const ϵ = 1e-15
boundedness(w, lb, ub) = all(lb - ϵ .≤ w .≤ ub + ϵ)
boundedness(w, lb, ub, var) = all(lb - ϵ .≤ getfield(w, var) .≤ ub + ϵ)
maximum_principle(w, w₀) = boundedness(w, minimum(w₀), maximum(w₀))
maximum_principle(w, w₀, var) = maximum_principle(getfield(w, var), getfield(w₀, var))

total_mass(mesh, w) = sum(w[i][:ρ] * FiniteVolumes.cell_volume(mesh, i) for i in nb_cells(mesh))
total_gas_mass(mesh, w) = sum(w[i][:ρ] * w[i][:ξ] * FiniteVolumes.cell_volume(mesh, i) for i in nb_cells(mesh))
mass_conservation(mesh, w, w₀) = (total_mass(mesh, w) == total_mass(mesh, w₀) && 
                                  total_gas_mass(mesh, w) == total_gas_mass(mesh, w₀))

riemann_problem(mesh, w₁, w₂) = [x[1] < 0.5 ? w₁ : w₂ for x in cell_centers(mesh)]

@testset "Cases" begin

@testset "Scalar problems" begin
    @testset "Trivial cases" begin
        mesh = RegularMesh1D(0.0, 1.0, 100)

        # No velocity => no change
        w₀ = rand(nb_cells(mesh))
        model = ScalarLinearAdvection(0.0)
        t, w = FiniteVolumes.run(model, mesh, w₀, time_step=0.001, nb_time_steps=3, verbose=false)
        @test all(w .== w₀)

        # Uniform initial condition => no change
        w₀ = ones(nb_cells(mesh))
        model = ScalarLinearAdvection(1.0)
        t, w = FiniteVolumes.run(model, mesh, w₀, time_step=0.001, nb_time_steps=3, verbose=false)
        @test all(w .== w₀)
    end

    @testset "1D linear advection" begin
        mesh = RegularMesh1D(0.0, 1.0, 100)

        sine_w₀ = map(x -> sin(x[1]), cell_centers(mesh))
        falling_step_w₀ = riemann_problem(mesh, 1.0, 0.0)
        rising_step_w₀ = riemann_problem(mesh, 0.0, 1.0)
        initial_conditions = [sine_w₀, falling_step_w₀, rising_step_w₀]

        forward_model = ScalarLinearAdvection(1.0)
        backward_model = ScalarLinearAdvection(-1.0)

        schemes = [Upwind()] #, Muscl(limiter=minmod), Muscl(limiter=ultrabee)]

        settings = (time_step=FixedCourant(0.1), nb_time_steps=5, verbose=false)

        for w₀ in initial_conditions, s in schemes
            # Test stability
            t, wf = FiniteVolumes.run(forward_model, mesh, w₀; numerical_flux=s, settings...)
            @test maximum_principle(wf, w₀)

            # Test left-right symmetry
            t, wb = FiniteVolumes.run(backward_model, mesh, reverse(w₀); numerical_flux=s, settings...)
            @test all(wf .== reverse(wb))

        end

        t, w_unstable = FiniteVolumes.run(forward_model, mesh, sine_w₀; time_step=FixedCourant(1.5), nb_time_steps=5, verbose=false)
        @test !maximum_principle(w_unstable, sine_w₀)
    end

    @testset "1D Burger" begin
        grid = RegularMesh1D(0.0, 1.0, 100)
        model = FiniteVolumes.AnonymousModel{Float64, 1}(u -> 0.5*u.^2)
        on_step(i) = nb_cells(grid)/3 < i < 2*nb_cells(grid)/3
        w₀ = [on_step(i) ? 1.0 : 0.0 for i in all_cells(grid)]
        t, w = FiniteVolumes.run(model, grid, w₀, time_step=FixedCourant(0.2), nb_time_steps=10, verbose=false)
        @test maximum_principle(w, w₀)
    end

    @testset "2D linear advection" begin
        grid = PeriodicRegularMesh2D(20, 20)
        model(u₀) = directional_splitting(ScalarLinearAdvection(u₀))

        square(x, side=0.5) = all(0.5-side/2 .<= x .<= 0.5+side/2) ? 1.0 : 0.0
        w₀ = map(square, cell_centers(grid))

        settings = (time_step=FixedCourant(0.1), nb_time_steps=5, verbose=false)

        schemes = [Upwind()] #, Muscl(limiter=minmod), Muscl(limiter=ultrabee), LagoutiereDownwind()]
        for s in schemes
            t, w = FiniteVolumes.run(model([1.0, 1.0]), grid, w₀; numerical_flux=s, settings...)
            @test maximum_principle(w, w₀)
        end
    end

    @testset "2D rotation" begin
        grid = RegularMesh2D(40, 40)
        u(x, center=(0.5, 0.5)) = [-(x[2]-center[2]), (x[1]-center[1])]
        model = FiniteVolumes.AnonymousModel{Float64, 2, true}((α, x) -> α .* u(x)) 

        square(x, side=0.5) = all(0.5-side/2 .<= x .<= 0.5+side/2) ? 1.0 : 0.0
        w₀ = map(square, cell_centers(grid))

        schemes = [Upwind()]
        settings = (time_step=FixedCourant(0.1), nb_time_steps=5, verbose=false)
        for s in schemes
            t, w = FiniteVolumes.run(model, grid, w₀; numerical_flux=s, settings...)
            @test maximum_principle(w, w₀)
        end
    end

    @testset "2D diagonal Burger" begin
        #   ∂_t α + u₀ ⋅ ∇ α^2/2 = 0
        # ⟺ ∂_t α + div(α^2/2 * u₀) = 0
        # ≡ rotated 1D Burger
        grid = RegularMesh2D(20, 20)
        model = FiniteVolumes.AnonymousModel{Float64, 2}(α -> 0.5*α.^2 * [1.0, 1.0])
        vertical_band(x) = 0.33 < x[1] < 0.66 ? 1.0 : 0.0
        w₀ = map(vertical_band, cell_centers(grid))
        t, w = FiniteVolumes.run(model, grid, w₀, time_step=FixedCourant(0.2), nb_time_steps=10, verbose=false)
        @test maximum_principle(w, w₀)
    end
end

@testset "N-scalar problems" begin
    @testset "2D linear advection" begin
        grid = PeriodicRegularMesh2D(-1.0, 1.0, 20, -1.0, 1.0, 20)
        model = directional_splitting(ScalarLinearAdvection(3, [1.0, 1.0]))

        function triple_point(x)
            if x[1] < 0.0
                return SVector(1.0, 0.0, 0.0)
            elseif x[2] < 0.0
                return SVector(0.0, 1.0, 0.0)
            else
                return SVector(0.0, 0.0, 1.0)
            end
        end
        w₀ = map(triple_point, cell_centers(grid))

        settings = (time_step=FixedCourant(0.2), nb_time_steps=10, verbose=false)

        schemes = [Upwind()] #, LagoutiereDownwind()]
        for s in schemes
            t, w = FiniteVolumes.run(model, grid, w₀; numerical_flux=s, settings...)
            @test maximum_principle(w, w₀, 1)
            @test maximum_principle(w, w₀, 2)
            @test maximum_principle(w, w₀, 3)
            @test all(sum.(w) .≈ 1.0)
        end
    end

end

@testset "Isothermal Euler problems" begin
    @testset "One-fluid shock tubes" begin
        mesh = RegularMesh1D(0.0, 1.0, 100)
        model = IsothermalTwoFluidEuler{nb_dims(mesh)}(1.0, 1.0, 5.0, 1000.0, 1.0)
        w₀ = riemann_problem(mesh, full_state(model, p=2.0, u=0.0, ξ=1.0), full_state(model, p=1.0, u=0.0, ξ=1.0))
        t, w = FiniteVolumes.run(model, mesh, w₀, time_step=FixedCourant(0.8), nb_time_steps=20, verbose=false)
        @test w[50][:p] ≈ 1.41  rtol=1e-2
        @test w[50][:u] ≈ 0.347 rtol=1e-2
        @test maximum_principle(w, w₀, :ξ)
        @test boundedness(w, 0.0, 1.0, :α)

        w₀ = riemann_problem(mesh, full_state(model, p=2.0, u=0.0, ξ=0.0), full_state(model, p=1.0, u=0.0, ξ=0.0))
        t, w = FiniteVolumes.run(model, mesh, w₀, time_step=FixedCourant(0.8), nb_time_steps=20, verbose=false)
        @test w[50][:p] ≈ 1.5  rtol=1e-2
        @test w[50][:u] ≈ 1e-4 rtol=1e-2
        @test maximum_principle(w, w₀, :ξ)
        @test boundedness(w, 0.0, 1.0, :α)
    end

    @testset "Two-fluid shock tubes" begin
        grid = RegularMesh1D(0.0, 1.0, 100)
        model = IsothermalTwoFluidEuler{nb_dims(grid)}(300.0, 1.0, 1500.0, 1000.0, 1e5)

        left_state = full_state(model, p=2e5, u=0.0, ξ=1.0)
        right_state = full_state(model, p=1e5, u=0.0, ξ=0.0)
        w₀ = [i < nb_cells(grid)/2 ? left_state : right_state for i in all_cells(grid)]

        t, w = FiniteVolumes.run(model, grid, w₀, time_step=FixedCourant(0.4), nb_time_steps=20, verbose=false)
        @test maximum_principle(w, w₀, :ξ)
        @test boundedness(w, 0.0, 1.0, :α)
    end

    @testset "Two-fluid 1D advection" begin
        grid = RegularMesh1D(0.0, 1.0, 100)
        model = IsothermalTwoFluidEuler{nb_dims(grid)}(300.0, 1.0, 1500.0, 1000.0, 1e5)

        ξ₀(x) = 0.2 < x[1] < 0.5 ? 0.0 : 1.0

        for u in [-1000.0, -10.0, 0.0, 10.0, 1000.0]
            w₀ = [full_state(model, p=1e5, u=u, ξ=ξ₀(x)) for x in cell_centers(grid)]
            t, w = FiniteVolumes.run(model, grid, w₀, time_step=FixedCourant(0.4), nb_time_steps=20, verbose=false)
            @test maximum_principle(w, w₀, :ξ)
            @test boundedness(w, 0.0, 1.0, :α)
            @test mass_conservation(grid, w, w₀)
            @test w[50][:u] ≈ u  atol=1e-10
            @test w[50][:p] ≈ 1e5
        end
    end

    @testset "Two-fluid 2D advection" begin
        grid = PeriodicRegularMesh2D(20, 20)
        model = IsothermalTwoFluidEuler{nb_dims(grid)}(300.0, 1.0, 1500.0, 1000.0, 1e5)

        is_in_disk(x) = norm([0.5, 0.5] - x) < 0.3
        ξ₀(x) = is_in_disk(x) ? 0.0 : 1.0

        for ux in [-10.0, 0.0, 10.0], uy in [-10.0, 0.0, 10.0]
            w₀ = [full_state(model, p=1e5, ux=ux, uy=uy, ξ=ξ₀(x)) for x in cell_centers(grid)]
            t, w = FiniteVolumes.run(model, grid, w₀, time_step=FixedCourant(0.3), nb_time_steps=20, verbose=false)
            @test maximum_principle(w, w₀, :ξ)
            @test boundedness(w, 0.0, 1.0, :α)
            @test mass_conservation(grid, w, w₀)
            @test w[50][:ux] ≈ ux  atol=1e-10
            @test w[50][:uy] ≈ uy  atol=1e-10
            @test w[50][:p] ≈ 1e5
        end
    end

end

end
