module FiniteVolumes

using LinearAlgebra
using StaticArrays

# CORE

include("./flux_function.jl")
export LinearAdvectionFlux, FluxFunction, ShallowWater
export directional_splitting

include("./scheme.jl")
export Upwind, NeumannBC

include("./courant.jl")

# EXAMPLE SUBMODULE
# Cartesian meshes
include("./cartesian_mesh/CartesianMeshes.jl")
using FiniteVolumes.CartesianMeshes
export CartesianMesh, PeriodicCartesianMesh, cell_centers

include("./splitting.jl")

# Time solver
include("./euler_explicit.jl")
export FixedCourant

# EXPERIMENTS
include("./diffusion.jl")
export DiffusionFlux

include("./experimental.jl")

end
