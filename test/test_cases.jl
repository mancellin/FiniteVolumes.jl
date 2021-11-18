# Integration tests on some simple test cases.

using Test
using StaticArrays, FiniteVolumes
using FiniteVolumes: UnidirectionalFlux
using FiniteVolumes.CartesianMeshes: Half

# TOOLS
getfield(w, var) = [wi[var] for wi in w]

const ϵ = 1e-15
boundedness(w, lb, ub) = all(lb - ϵ .≤ w .≤ ub + ϵ)
boundedness(w, lb, ub, var) = all(lb - ϵ .≤ getfield(w, var) .≤ ub + ϵ)
maximum_principle(w, w₀) = boundedness(w, minimum(w₀), maximum(w₀))
maximum_principle(w, w₀, var) = maximum_principle(getfield(w, var), getfield(w₀, var))

riemann_problem(mesh, w₁, w₂, step_position=0.5) = [x[1] < step_position ? w₁ : w₂ for x in cell_centers(mesh)]

@testset "Cases" begin

@testset "Scalar problems" begin
    @testset "Trivial cases" begin
        mesh = CartesianMesh(10)

        # No velocity => no change
        w₀ = rand(FiniteVolumes.nb_cells(mesh))
        flux = LinearAdvectionFlux(0.0)
        t, w = FiniteVolumes.run(flux, mesh, w₀, time_step=0.001, nb_time_steps=3, verbose=false)
        @test all(w .== w₀)

        # Uniform initial condition => no change
        w₀ = ones(FiniteVolumes.nb_cells(mesh))
        flux = LinearAdvectionFlux(1.0)
        t, w = FiniteVolumes.run(flux, mesh, w₀, time_step=0.001, nb_time_steps=3, verbose=false)
        @test all(w .== w₀)
    end

    @testset "1D linear advection" begin
        mesh = CartesianMesh(100)

        sine_w₀ = map(x -> sin(x[1]), cell_centers(mesh))
        falling_step_w₀ = riemann_problem(mesh, 1.0, 0.0)
        rising_step_w₀ = riemann_problem(mesh, 0.0, 1.0)
        initial_conditions = [sine_w₀, falling_step_w₀, rising_step_w₀]

        forward_flux = LinearAdvectionFlux(1.0)
        backward_flux = LinearAdvectionFlux(-1.0)

        settings = (time_step=FixedCourant(0.1), nb_time_steps=5, verbose=false, numerical_flux=Upwind())

        for w₀ in initial_conditions
            # Test stability
            t, wf = FiniteVolumes.run(forward_flux, mesh, w₀; settings...)
            @test maximum_principle(wf, w₀)

            # Test left-right symmetry
            t, wb = FiniteVolumes.run(backward_flux, mesh, reverse(w₀); settings...)
            @test all(wf .== reverse(wb))

        end

        t, w_unstable = FiniteVolumes.run(forward_flux, mesh, sine_w₀; time_step=FixedCourant(1.5), nb_time_steps=5, verbose=false)
        @test !maximum_principle(w_unstable, sine_w₀)
    end

    @testset "1D Wave equation" begin
        mesh = CartesianMesh(10)
        w₀ = [SVector{2}(sin(2π*x), cos(2π*x)) for x in cell_centers(mesh)]

        # Anonymous model
        f(α, n) = n * SVector(α[2], α[1])
        t, w₁ = FiniteVolumes.run(f, mesh, w₀, time_step=FixedCourant(0.2), nb_time_steps=10, verbose=false)

        # Fancy model
        struct Wave1DFlux{T}
            velocity::T
        end
        (wf::Wave1DFlux)(w, n) = wf.velocity*SVector{2}(w[2], w[1])*n
        jacobian(f::Wave1DFlux, w, n) = f.velocity*SMatrix{2, 2}(0.0, 1.0, 1.0, 0.0)*n
        FiniteVolumes.eigvals(f::Wave1DFlux, w, n) = SVector{2}(-1.0, 1.0)*n
        FiniteVolumes.eigen(f::Wave1DFlux, w, n) = (SVector{2}(-1.0, 1.0)*n, SMatrix{2, 2}(-0.707107, 0.707107, 0.707107, 0.707107)*n)
        t, w₂ = FiniteVolumes.run(Wave1DFlux(1.0), mesh, w₀, time_step=FixedCourant(0.2), nb_time_steps=10, verbose=false)

        @test all(isapprox.(w₁, w₂, atol=1e-6))
    end

    @testset "1D Burger" begin
        mesh = CartesianMesh(100)
        flux(u, n) = 0.5*u.^2 * n
        step(x) = 0.33 < x[1] < 0.66 ? 1.0 : 0.0
        w₀ = map(step, cell_centers(mesh))
        t, w = FiniteVolumes.run(flux, mesh, w₀, time_step=FixedCourant(0.2), nb_time_steps=10, verbose=false)
        @test maximum_principle(w, w₀)
    end

    @testset "Time dependant advection" begin
        mesh = CartesianMesh(100)
        flux = FiniteVolumes.TimeDependantFlux(t -> LinearAdvectionFlux(sin(t)))
        step(x) = 0.33 < x[1] < 0.66 ? 1.0 : 0.0
        w₀ = map(step, cell_centers(mesh))
        t, w = FiniteVolumes.run(flux, mesh, w₀, time_step=0.005, nb_time_steps=10, verbose=false)
        @test maximum_principle(w, w₀)
    end

    @testset "Splitted 2D linear advection" begin
        mesh = PeriodicCartesianMesh(20, 20)
        splitted_advection(u₀) = FiniteVolumes.directional_splitting(LinearAdvectionFlux(u₀))

        @test splitted_advection(SVector(1.0, 1.0)) == (UnidirectionalFlux{1}(LinearAdvectionFlux(SVector(1.0, 0.0))), UnidirectionalFlux{2}(LinearAdvectionFlux(SVector(0.0, 1.0))))

        settings = (time_step=FixedCourant(0.1), nb_time_steps=5, numerical_flux=Upwind(), verbose=false)

        # Scalar
        square(x, side=0.5) = all(0.5-side/2 .<= x .<= 0.5+side/2) ? 1.0 : 0.0
        w₀ = map(square, cell_centers(mesh))
        t, w = FiniteVolumes.run(splitted_advection(SVector(1.0, 1.0)), mesh, w₀; settings...)
        @test maximum_principle(w, w₀)

        # Multi-scalar
        function triple_point(x)
            if x[1] < 0.5
                return SVector(1.0, 0.0, 0.0)
            elseif x[2] < 0.5
                return SVector(0.0, 1.0, 0.0)
            else
                return SVector(0.0, 0.0, 1.0)
            end
        end
        w₀ = map(triple_point, cell_centers(mesh))
        t, w = FiniteVolumes.run(splitted_advection(SVector(1.0, 1.0)), mesh, w₀; settings...)
        @test maximum_principle(w, w₀, 1)
        @test maximum_principle(w, w₀, 2)
        @test maximum_principle(w, w₀, 3)
        @test all(sum.(w) .≈ 1.0)
    end

    @testset "2D rotation" begin
        mesh = PeriodicCartesianMesh(40, 40)
        flux = RotationFlux(SVector(0.5, 0.5))

        square(x, side=0.5) = all(0.5-side/2 .<= x .<= 0.5+side/2) ? 1.0 : 0.0
        w₀ = map(square, cell_centers(mesh))

        settings = (time_step=FixedCourant(0.1), nb_time_steps=10, numerical_flux=Upwind(), verbose=false)
        t, w = FiniteVolumes.run(flux, mesh, w₀; settings...)
        @test maximum_principle(w, w₀)

        t, w = FiniteVolumes.run(directional_splitting(flux), mesh, w₀; settings...)
        @test maximum_principle(w, w₀)
    end

    @testset "2D diagonal Burger" begin
        #   ∂_t α + u₀ ⋅ ∇ α^2/2 = 0
        # ⟺ ∂_t α + div(α^2/2 * u₀) = 0
        # ≡ rotated 1D Burger
        mesh = PeriodicCartesianMesh(20, 20)
        flux(α, n) = 0.5*α.^2 * n' * [1.0, 1.0]
        vertical_band(x) = 0.33 < x[1] < 0.66 ? 1.0 : 0.0
        w₀ = map(vertical_band, cell_centers(mesh))
        t, w = FiniteVolumes.run(flux, mesh, w₀, time_step=FixedCourant(0.2), nb_time_steps=10, verbose=false)
        @test maximum_principle(w, w₀)
    end

    @testset "Splitted 3D linear advection" begin
        mesh = PeriodicCartesianMesh(20, 20, 20)
        splitted_advection(u₀) = FiniteVolumes.directional_splitting(LinearAdvectionFlux(u₀))

        settings = (time_step=FixedCourant(0.1), nb_time_steps=5, numerical_flux=Upwind(), verbose=false)

        # Scalar
        cube(x, side=0.5) = all(0.5-side/2 .<= x .<= 0.5+side/2) ? 1.0 : 0.0
        w₀ = map(cube, cell_centers(mesh))
        t, w = FiniteVolumes.run(splitted_advection([1.0, 1.0, 1.0]), mesh, w₀; settings...)
        @test maximum_principle(w, w₀)
    end
end

@testset "Shallow water" begin

    total_mass(mesh, v) = sum(v[i][1] * FiniteVolumes.cell_volume(mesh, i) for i in FiniteVolumes.all_cells(mesh))
    total_momentum(mesh, v) = sum(v[i][2] * FiniteVolumes.cell_volume(mesh, i) for i in FiniteVolumes.all_cells(mesh))

    @testset "1D" begin
        flux = ShallowWater(9.8)
        @test Upwind()(flux, CartesianMesh(2), [SVector(1.0, 0.0), SVector(1.0, 0.0)], (Half(3,))) == SVector(0.0, 9.8/2)
        @test Upwind()(flux, CartesianMesh(2), [SVector(2.0, 0.0), SVector(1.0, 0.0)], (Half(3,))) ≈ SVector(1.9170289512680811, 12.25)

        mesh = CartesianMesh(100)
        w₀ = riemann_problem(mesh, SVector(2.0, 0.0), SVector(1.0, 0.0))
        t, w = FiniteVolumes.run(flux, mesh, w₀, time_step=FixedCourant(0.4),
                                 nb_time_steps=50, verbose=false)

        w₀ = map(x -> SVector(1.0 + exp(-500*(x[1]-0.5)^2), 0.0), cell_centers(mesh))
        t, w = FiniteVolumes.run(flux, mesh, w₀, time_step=FixedCourant(0.4),
                                 nb_time_steps=50, verbose=false)

        @test total_mass(mesh, w₀) .≈ total_mass(mesh, w)
        @test total_momentum(mesh, w₀) .≈ total_momentum(mesh, w) atol=1e-15
        # plot(mesh, w, 1); plot!(mesh, w, 2)
    end

    @testset "2D" begin
        mesh = PeriodicCartesianMesh(40, 40)
        flux = ShallowWater(9.8)

        v₀ = map(x -> SVector(1.0 + exp(-500*((x[1]-0.5)^2 + (x[2]-0.5)^2)), 0.0, 0.0), cell_centers(mesh))
        FiniteVolumes.div(flux, mesh, v₀, Upwind())
        t, v = FiniteVolumes.run(flux, mesh, v₀, time_step=FixedCourant(0.3),
                                 nb_time_steps=40, verbose=false)

        @test total_mass(mesh, v₀) .≈ total_mass(mesh, v)
        @test total_momentum(mesh, v₀) .≈ total_momentum(mesh, v) atol=1e-15
        # plot(mesh, v, 1)
    end

end

end
