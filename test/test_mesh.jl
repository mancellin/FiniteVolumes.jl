using Test

using StaticArrays
using LinearAlgebra: norm
using FiniteVolumes

grid = PeriodicRegularMesh2D(3, 3)
st = FiniteVolumes.stencil(grid, 5)
@test st[-1, -1] == 1
@test st[1, 0] == 6
@test st[1, 1] == 9

st = FiniteVolumes.stencil(grid, 1)
@test st[-1, -1] == 9
@test st[1, 0] == 2
@test st[1, 1] == 5

st = FiniteVolumes.stencil(grid, 3)
@test st[-1, -1] == 8
@test st[1, 0] == 1
@test st[1, 1] == 4

methods = [
	FiniteVolumes.central_differences_gradient,
	FiniteVolumes.youngs_gradient,
	FiniteVolumes.least_square_gradient,
]

horizontal = Float64[0, 0, 0, 1/2, 1/2, 1/2, 1, 1, 1]
for f in methods
	g = f(grid, horizontal, 5)
	@test all(g .≈ [0.0, 3/2])
end

diagonal = Float64[-1/2, 0, 1/2, 0, 1/2, 1, 1/2, 1, 3/2]
for f in methods
	g = f(grid, diagonal, 5)
	@test all(g .≈ [3/2, 3/2])
end

interface_1 = [1, 1, 1, 1/6, 1/2, 5/6, 0, 0, 0]
g = FiniteVolumes.youngs_gradient(grid, interface_1, 5)
@test -g[1]/g[2] ≈ 1/3

#= g = FiniteVolumes.youngs_gradient(grid, [1, 1, 1, 1/3, 11/12, 1, 0, 1/12, 2/3], 5) =#
#= @test -g[1]/g[2] ≈ 2/3 =#


gridx, gridy = directional_splitting(grid)
@test FiniteVolumes.left_gradient(gridx, horizontal, 5) ≈ 0.0
@test FiniteVolumes.right_gradient(gridx, horizontal, 5) ≈ 0.0
@test FiniteVolumes.left_gradient(gridy, horizontal, 5) ≈ 1.5
@test FiniteVolumes.right_gradient(gridy, horizontal, 5) ≈ 1.5

