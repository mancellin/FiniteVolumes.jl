using Test
using FiniteVolumes: ScalarLinearAdvection, local_upwind_flux
using StaticArrays

from_left = ScalarLinearAdvection(1.0)
from_right = ScalarLinearAdvection(-1.0)

@test local_upwind_flux(from_left, 1.0, [], 0.0, []) == (1.0, 1.0)
@test local_upwind_flux(from_right, 1.0, [], 0.0, []) == (0.0, 1.0)

@test local_upwind_flux(from_left, (1.0,), [], (0.0,), []) == ((1.0,), 1.0)
@test local_upwind_flux(from_right, (1.0,), [], (0.0,), []) == ((0.0,), 1.0)

from_left_2d = ScalarLinearAdvection([1.0, 0.0])
from_right_2d = ScalarLinearAdvection([-1.0, 0.0])

@test local_upwind_flux(from_left_2d, 1.0, [], 0.0, []) == (1.0, 1.0)
@test local_upwind_flux(from_right_2d, 1.0, [], 0.0, []) == (0.0, 1.0)
