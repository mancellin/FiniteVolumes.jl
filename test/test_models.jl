# Tests of the AbstractModel objects in FiniteVolumes.jl

using Test
using StaticArrays
using FiniteVolumes

@testset "Models" begin

@testset "Anonynous models" begin
    @testset "(1 var, 1D)" begin
        flux = FluxFunction{Float64, 1}(α -> 0.5*α.^2)
        @test flux(3.0, 1.0) == 4.5
        @test FiniteVolumes.jacobian(flux, 2.0, 1.0) == 2.0
    end

    @testset "(1 var, 2D)" begin
        flux = FluxFunction{Float64, 2}(α -> α .* [α, 1.0])
        @test flux(1.0, SVector(1.0, 0.0)) == 1.0
        @test flux(1.0, SVector(0.0, 1.0)) == 1.0
        @test FiniteVolumes.jacobian(flux, 1.0, SVector(1.0, 0.0)) == 2.0
        @test FiniteVolumes.jacobian(flux, 1.0, SVector(0.0, 1.0)) == 1.0
    end

    @testset "(2 var, 1D)" begin
        flux = FluxFunction{SVector{2, Float64}, 1}(u -> [0.5*u[1].^2 + u[2], u[2]])
        @test flux([1.0, 2.0], 1.0) == [2.5, 2.0]
        @test flux(SVector(1.0, 2.0), 1.0) == SVector(2.5, 2.0)
        @test FiniteVolumes.jacobian(flux, [1.0, 1.0], 1.0) == [1.0 1.0; 0.0 1.0]
    end

    @testset "(2 var, 2D)" begin
        flux = FluxFunction{SVector{2, Float64}, 2}(u -> [[u[1], u[2]], [0.0, 0.0]])
        @test flux([1.0, 2.0], [1.0, 0.0]) == [1.0, 2.0]
        @test flux([1.0, 2.0], [0.0, 1.0]) == [0.0, 0.0]
        @test FiniteVolumes.jacobian(flux, [1.0, 2.0], [1.0, 0.0]) == [1.0 0.0; 0.0 1.0]
        @test FiniteVolumes.jacobian(flux, [1.0, 2.0], [0.0, 1.0]) == [0.0 0.0; 0.0 0.0]
    end
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
    end

    @testset "2D" begin
        f = ShallowWater()
        for h in [1.0, 2.0], ux in [-1.0, 0.0, 1.0], uy = [0.0, 1.0]
            @test all(FiniteVolumes.eigvals(f, SVector(h, h*ux, h*uy), SVector(1.0, 0.0)) .≈ [ux-sqrt(h*f.g), ux, ux+sqrt(h*f.g)])
        end
    end
end

end
