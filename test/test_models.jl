using Test
using StaticArrays
using FiniteVolumes


@testset "Anonynous models" begin
    @testset "(1 var, 1D)" begin
        model = FiniteVolumes.AnonymousModel{1, 1, Float64}(α -> 0.5*α.^2)
        @test FiniteVolumes.normal_flux(model, 3.0) == 4.5
        @test FiniteVolumes.jacobian(model, 2.0) == 2.0
    end

    @testset "(2 var, 1D)" begin
        model = FiniteVolumes.AnonymousModel{2, 1, Float64}(u -> [0.5*u[1].^2 + u[2], u[2]])
        @test FiniteVolumes.jacobian(model, SVector{2, Float64}(1.0, 1.0)) == [1.0 1.0; 0.0 1.0]
    end

    @testset "(1 var, 2D)" begin
        model = FiniteVolumes.AnonymousModel{1, 2, Float64}(α -> α .* [α, 1.0])
        @test FiniteVolumes.normal_flux(FiniteVolumes.rotate_model(model, [1 0; 0 1]), 1.0) == 1.0
        @test FiniteVolumes.jacobian(FiniteVolumes.rotate_model(model, [1 0; 0 1]), 1.0) == 2.0
        @test FiniteVolumes.normal_flux(FiniteVolumes.rotate_model(model, [0 1; 1 0]), 1.0) == 1.0
        @test FiniteVolumes.jacobian(FiniteVolumes.rotate_model(model, [0 1; 1 0]), 1.0) == 1.0
    end

    @testset "(2 var, 2D)" begin
        model = FiniteVolumes.AnonymousModel{2, 2, Float64}(u -> [
                                                                  [u[1], u[2]],
                                                                  [0.0, 0.0],
                                                                 ])
        w = @SVector [1.0, 2.0]
        @test FiniteVolumes.normal_flux(FiniteVolumes.rotate_model(model, [1 0; 0 1]), w) == w
        @test FiniteVolumes.jacobian(FiniteVolumes.rotate_model(model, [1 0; 0 1]), w) == [1.0 0.0; 0.0 1.0]
        @test FiniteVolumes.normal_flux(FiniteVolumes.rotate_model(model, [0 1; 1 0]), w) == 0.0*w
        @test FiniteVolumes.jacobian(FiniteVolumes.rotate_model(model, [0 1; 1 0]), w) == [0.0 0.0; 0.0 0.0]
    end
end

