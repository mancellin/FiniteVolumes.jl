#!/usr/bin/env julia

using Test

@testset "FiniteVolumes.jl" begin
    include("test_mesh.jl")
    include("test_stencil.jl")
    include("test_scheme.jl")
    include("test_cases.jl")
end
