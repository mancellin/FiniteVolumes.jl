# Tests of the AbstractModel objects in FiniteVolumes.jl

using Test
using StaticArrays
using FiniteVolumes
using ForwardDiff, LinearAlgebra

@testset "Fluxes" begin

@testset "Anonynous models" begin
    @testset "(1 var, 1D)" begin
        flux(α, n) = 0.5*α.^2 * n
        @test flux(3.0, 1.0) == 4.5
        @test FiniteVolumes.jacobian(flux, 2.0, 1.0) == 2.0
    end

    @testset "(1 var, 2D)" begin
        flux(α, n) = n' * [α, 1.0] * α
        @test flux(1.0, SVector(1.0, 0.0)) == 1.0
        @test flux(1.0, SVector(0.0, 1.0)) == 1.0
        @test FiniteVolumes.jacobian(flux, 1.0, SVector(1.0, 0.0)) == 2.0
        @test FiniteVolumes.jacobian(flux, 1.0, SVector(0.0, 1.0)) == 1.0
    end

    @testset "(2 var, 1D)" begin
        flux(u, n) = [0.5*u[1].^2 + u[2], u[2]] .* n
        @test flux([1.0, 2.0], 1.0) == [2.5, 2.0]
        @test flux(SVector(1.0, 2.0), 1.0) == SVector(2.5, 2.0)
        @test FiniteVolumes.jacobian(flux, [1.0, 1.0], 1.0) == [1.0 1.0; 0.0 1.0]
    end

    @testset "(2 var, 2D)" begin
        flux(u, n) = n' * [[u[1], u[2]], [0.0, 0.0]]
        @test flux([1.0, 2.0], [1.0, 0.0]) == [1.0, 2.0]
        @test flux([1.0, 2.0], [0.0, 1.0]) == [0.0, 0.0]
        @test FiniteVolumes.jacobian(flux, [1.0, 2.0], [1.0, 0.0]) == [1.0 0.0; 0.0 1.0]
        @test FiniteVolumes.jacobian(flux, [1.0, 2.0], [0.0, 1.0]) == [0.0 0.0; 0.0 0.0]
    end
end

@testset "LinearAdvection" begin
    mesh = CartesianMesh(10)
    w0 = [sin(2π*x) for x in cell_centers(mesh)]
    i_face = (FiniteVolumes.CartesianMeshes.Half(11),)

    flux = LinearAdvectionFlux(1.0)
    @test (Upwind())(flux, mesh, w0, i_face) == w0[5]
    # @btime (Upwind())($f, $mesh, $w, $i_face)  # ~4ns

    flux = LinearAdvectionFlux(-1.0)
    @test (Upwind())(flux, mesh, w0, i_face) == -w0[6]
end

@testset "Shallow water" begin
    @testset "1D" begin
        f = ShallowWater()
        @test f(SVector(1.0, 0.0), 1.0) == SVector(0.0, 9.81/2)
        @test f(SVector(1.0, 1.0), 1.0) == SVector(1.0, 1 + 9.81/2)
        @test f(SVector(1.0, 1.0), -1.0) == -SVector(-1.0, 1 + 9.81/2)

        for h in [1.0, 2.0], u in [-1.0, 0.0, 1.0]
            @test all(FiniteVolumes.eigvals(f, SVector(h, h*u), 1.0) .≈ SVector(u-sqrt(h*f.g), u+sqrt(h*f.g)))
        end

        f2(v, n) = n * SVector(v[2], v[2]^2/v[1] + v[1]^2*9.81/2)
        mesh = CartesianMesh(2)
        w = [SVector(1.0, 0.0), SVector(2.0, 0.0)]
        i_face = (FiniteVolumes.CartesianMeshes.Half(3),)
        @test (Upwind())(f, mesh, w, i_face) ≈ (Upwind())(f2, mesh, w, i_face)
    end

    @testset "2D" begin
        flux = ShallowWater(9.8)

        for v in [SVector(1.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0)], n in [SVector(1.0, 0.0), SVector(0.0, 1.0), sqrt(2)/2*SVector(1.0, 1.0)]
            J1 = ForwardDiff.jacobian(v -> flux(v, n), v)
            J2 = FiniteVolumes.jacobian(flux, v, n)
            @test J1 == J2
            λ1 = LinearAlgebra.eigvals(Array(J1))
            λ2 = FiniteVolumes.eigvals(flux, v, n)
            @test λ1 ≈ λ2
            eg1 = LinearAlgebra.eigen(Array(J1)).vectors
            eg2 = FiniteVolumes.eigen(flux, v, n)[2]
            for (c1, c2) in zip(eachcol(eg1), eachcol(eg2))
                @test maximum(abs.(c2)) .* abs.(c1) ≈ maximum(abs.(c1)) .* abs.(c2)
            end
        end
    end
end

end
