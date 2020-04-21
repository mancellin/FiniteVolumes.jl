using Test
using FiniteVolumes
using FiniteVolumes: local_upwind_flux
using StaticArrays

from_left = ScalarLinearAdvection(1.0)
faster_from_left = ScalarLinearAdvection(10.0)
from_right = ScalarLinearAdvection(-1.0)

@test local_upwind_flux(from_left, 1.0, [], 0.0, []) == 1.0
@test local_upwind_flux(from_right, 1.0, [], 0.0, []) == 0.0
@test local_upwind_flux(from_left, SVector(1.0), [], SVector(0.0), []) == SVector(1.0)
@test local_upwind_flux(from_right, SVector(1.0), [], SVector(0.0), []) == SVector(0.0)

from_left_2d = ScalarLinearAdvection([1.0, 0.0])
from_right_2d = ScalarLinearAdvection([-1.0, 0.0])

@test local_upwind_flux(from_left_2d, 1.0, [], 0.0, []) == 1.0
@test local_upwind_flux(from_right_2d, 1.0, [], 0.0, []) == 0.0
@test local_upwind_flux(from_left_2d, SVector(1.0), [], SVector(0.0), []) == SVector(1.0)
@test local_upwind_flux(from_right_2d, SVector(1.0), [], SVector(0.0), []) == SVector(0.0)

from_left_2f_2d = ScalarLinearAdvection(2, [1.0, 0.0])
from_right_2f_2d = ScalarLinearAdvection(2, [-1.0, 0.0])

@test local_upwind_flux(from_left_2f_2d, SVector(1.0, 1.0), [], SVector(0.0, 0.0), []) == SVector(1.0, 1.0)
@test local_upwind_flux(from_right_2f_2d, SVector(1.0, 1.0), [], SVector(0.0, 0.0), []) == SVector(0.0, 0.0)


# Courant

mesh = RegularMesh1D(0.0, 2.0, 2)
@test FiniteVolumes.courant(1.0, mesh, from_left, [1.0, 1.0], [[], []]) == 1.0
@test FiniteVolumes.courant(1.0, mesh, faster_from_left, [1.0, 1.0], [[], []]) == 10.0
@test FiniteVolumes.courant(1.0, mesh, from_right, [1.0, 1.0], [[], []]) == 1.0

