module FiniteVolumes

using ForwardDiff
using LinearAlgebra
using StaticArrays

# CORE

include("./flux_function.jl")
export LinearAdvectionFlux, AdvectionFlux, RotationFlux, ShallowWater

include("./scheme.jl")
export Upwind, Centered, NeumannBC

include("./courant.jl")

# EXAMPLE SUBMODULE
# Cartesian meshes
include("./cartesian_mesh/CartesianMeshes.jl")
using FiniteVolumes.CartesianMeshes
export CartesianMesh, PeriodicCartesianMesh, cell_centers

include("./splitting.jl")
export directional_splitting

# Time solver
include("./euler_explicit.jl")
export FixedCourant

# EXPERIMENTS
include("./diffusion.jl")
export DiffusionFlux

include("./experimental.jl")

end
