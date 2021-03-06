#!/usr/bin/env julia

using Test

@testset "FiniteVolumes.jl" begin
    # Unit tests
    include("test_mesh.jl")
    include("test_flux_functions.jl")
    include("test_scheme.jl")

    # Integration tests
    include("test_cases.jl")
    include("test_interface.jl")
end
