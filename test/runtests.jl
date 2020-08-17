#!/usr/bin/env julia

using Test

@testset "FiniteVolumes.jl" begin
    # Unit tests
    include("test_mesh.jl")
    include("test_stencil.jl")
    include("test_models.jl")
    include("test_scheme.jl")

    # Integration tests
    include("test_cases.jl")
end
