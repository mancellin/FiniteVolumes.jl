# A minimalistic generic Julia toolkit for the Finite Volume Method

## Installation

```
]add https://github.com/mancellin/FiniteVolumes.jl
```

## Presentation

The main interface of this package is the `FiniteVolumes.div` function, meant to compute the discrete divergence operator with a Finite Volume scheme on a given mesh.

```julia
using FiniteVolumes
mesh = CartesianMesh(10, 10)  # A 2D mesh of 10 cells × 10 cells
F(u) = u * [1.0, 0.5]  # Flux function of the linear advection flux in direction (1.0, 0.5)
dudt(U) = -FiniteVolumes.div(F, mesh, U, (Upwind(), NeumannBC()))

U = [sin(2π*i/10*j/10) for i in 1:10, j in 1:10]  # A scalar field
# A basic explicit Euler time step to solve the advection PDE ∂_t u + div (c u) = 0
Δt = 0.01
U += Δt * dudt(U)
```

## Philosophy

The library is *minimalistic*, in the sense that it only implements the least common denominator of most Finite Volume methods.
The number of meshes and methods directly provided is very limited, but the library is also *extensible*, in the sense that it should be easy to add new methods or use methods defined in other packages.

As an example, the package only include a basic explicit Euler integrator for time integration, but can be coupled with `OrdinaryDiffEq.jl` for more advanced methods.


## Other packages of interest for Finite Volume Method in Julia

* https://github.com/trixi-framework/Trixi.jl
* https://github.com/j-fu/VoronoiFVM.jl
* https://github.com/vavrines/Kinetic.jl
* https://github.com/CliMA/Oceananigans.jl
