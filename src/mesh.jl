using OffsetArrays

abstract type FiniteVolumeMesh end

################################################################################
#                                      1D                                      #
################################################################################

struct RegularMesh1D <: FiniteVolumeMesh
    x_min::Float64
    x_max::Float64
    nb_cells::Int64
end

@inline dx(grid::RegularMesh1D) = (grid.x_max - grid.x_min)/grid.nb_cells

@inline nb_dims(grid::RegularMesh1D) = 1

@inline nb_cells(grid::RegularMesh1D) = grid.nb_cells
@inline nb_faces(grid::RegularMesh1D) = grid.nb_cells + 1

cell_center(grid::RegularMesh1D, i_cell) = SVector{1, Float64}(dx(grid)*(Float64(i_cell) - 0.5))
face_center(grid::RegularMesh1D, i_face) = SVector{1, Float64}(dx(grid)*(Float64(i_face) - 1.0))

@inline inner_faces(grid::RegularMesh1D) = Vector(2:(nb_faces(grid)-1))
@inline cells_next_to_inner_face(grid::RegularMesh1D, i_face)::Tuple{Int64, Int64} = (i_face-1, i_face)

@inline boundary_faces(grid::RegularMesh1D) = [1, nb_faces(grid)]
@inline cell_next_to_boundary_face(grid::RegularMesh1D, i_face)::Int64 = i_face == 1 ? 1 : nb_cells(grid)

@inline face_area(grid::RegularMesh1D, i_face)::Float64 = 1.0
@inline cell_volume(grid::RegularMesh1D, i_cell)::Float64 = dx(grid)

@inline rotation_matrix(grid::RegularMesh1D, i_face) = SMatrix{1, 1, Float64}(i_face == 1 ? -1.0 : 1.0)

# Indices of the cells in the 3 stencil around i_cell
function stencil(grid::RegularMesh1D, i_cell)
    left_cell = i_cell == 1 ? 1 : i_cell - 1
    right_cell = i_cell == nb_cells(grid) ? nb_cells(grid) : i_cell + 1
    stencil = SVector{3, Int}(left_cell, i_cell, right_cell)
    return OffsetArray(stencil, -1:1)
end

function oriented_stencil(mesh::RegularMesh1D, i_cell, i_face)
    st = stencil(mesh, i_cell)
    if face_center_relative_to_cell(mesh, i_cell, i_face)' * rotation_matrix(mesh, i_face)[:, 1] > 0
        return st
    else
        return reverse(st)
    end
end

function face_center_relative_to_cell(grid::RegularMesh1D, i_cell, i_face)
	face_center(grid, i_face) - cell_center(grid, i_cell)
end

################################################################################
#                                      2D                                      #
################################################################################

abstract type AbstractRegularMesh2D <: FiniteVolumeMesh end

struct RegularMesh2D <: AbstractRegularMesh2D
    x_min::Float64
    x_max::Float64
    nx::Int64
    y_min::Float64
    y_max::Float64
    ny::Int64
end

RegularMesh2D(nx::Int, ny::Int) = RegularMesh2D(0.0, 1.0, nx, 0.0, 1.0, ny)

struct PeriodicRegularMesh2D <: AbstractRegularMesh2D
    x_min::Float64
    x_max::Float64
    nx::Int64
    y_min::Float64
    y_max::Float64
    ny::Int64
end

PeriodicRegularMesh2D(nx::Int, ny::Int) = PeriodicRegularMesh2D(0.0, 1.0, nx, 0.0, 1.0, ny)

nb_dims(grid::AbstractRegularMesh2D) = 2

nb_cells(grid::AbstractRegularMesh2D) = grid.nx * grid.ny

nb_faces(grid::RegularMesh2D) = 2 * grid.nx * grid.ny + grid.nx + grid.ny
nb_faces(grid::PeriodicRegularMesh2D) = 2 * grid.nx * grid.ny

dx(grid::AbstractRegularMesh2D) = (grid.x_max - grid.x_min)/grid.nx
dy(grid::AbstractRegularMesh2D) = (grid.y_max - grid.y_min)/grid.ny

function cell_center(grid::AbstractRegularMesh2D, i_cell)
    SVector{2, Float64}(
                        grid.x_min + dx(grid)*((i_cell - 1) % grid.nx + 0.5),
                        grid.x_min + dy(grid)*((i_cell - 1) ÷ grid.nx + 0.5)
                       )
end

# By convention, all the horizontal faces have even indices.
_is_horizontal(i_face) = i_face % 2 == 0

_is_on_right_edge(m::RegularMesh2D, i_face) = (i_face <= 2*m.ny*m.nx) && (i_face % (2*m.nx) == 2*m.nx - 1)
_is_on_top_edge(m::RegularMesh2D, i_face) = ((i_face-1) ÷ (2*m.nx) == (m.ny-1)) && (_is_horizontal(i_face))

_is_on_left_edge(m::RegularMesh2D, i_face) = (i_face > 2*m.ny*m.nx) && (!_is_horizontal(i_face))
_is_on_bottom_edge(m::RegularMesh2D, i_face) = (i_face > 2*m.ny*m.nx) && (_is_horizontal(i_face))

function _is_inner_face(m::RegularMesh2D, i_face)
    (i_face <= 2*m.ny*m.nx) && !_is_on_right_edge(m, i_face) && !_is_on_top_edge(m, i_face)
end
inner_faces(grid::RegularMesh2D) = Iterators.filter(i -> _is_inner_face(grid, i), 1:(nb_faces(grid)))
boundary_faces(grid::RegularMesh2D) = Iterators.filter(i -> !_is_inner_face(grid, i), 1:(nb_faces(grid)))

inner_faces(grid::PeriodicRegularMesh2D) = 1:(nb_faces(grid))
boundary_faces(grid::PeriodicRegularMesh2D) = []

function cells_next_to_inner_face(grid::AbstractRegularMesh2D, i_face)
    # ! No check that the face is really an inner face !
    # Horizontal face
    if _is_horizontal(i_face) 
        i_cell_1 = i_face ÷ 2
        return (i_cell_1, (i_cell_1 + grid.nx -1) % nb_cells(grid) + 1)

    # Vertical face
    else
        i_cell_1 = (i_face + 1) ÷ 2
        if i_cell_1 % grid.nx == 0
            return (i_cell_1, i_cell_1 + 1 - grid.nx)  # Last vertical face on the right
        else
            return (i_cell_1, i_cell_1 + 1)
        end
    end
end

function cell_next_to_boundary_face(grid::RegularMesh2D, i_face)
    if _is_on_top_edge(grid, i_face)
        return i_face ÷ 2
    elseif _is_on_bottom_edge(grid, i_face)
        return (i_face ÷ 2 - 1) % grid.nx + 1
    elseif _is_on_right_edge(grid, i_face)
        return (i_face + 1) ÷ 2
    elseif _is_on_left_edge(grid, i_face)
        return ((i_face - 2*grid.nx*grid.ny - 1) ÷ 2) * grid.nx + 1
    else
        error("This should not happen.")
    end
end
cell_next_to_boundary_face(grid::PeriodicRegularMesh2D, i_face) = error("No boundary faces in this mesh.")



face_area(grid::AbstractRegularMesh2D, i_face)::Float64 = _is_horizontal(i_face) ? dx(grid) : dy(grid)

cell_volume(grid::AbstractRegularMesh2D, i_cell)::Float64 = dx(grid)*dy(grid)

_bottom_left_corner(grid::AbstractRegularMesh2D, i_cell) = cell_center(grid, i_cell) .- @SVector [dx(grid)/2, dy(grid)/2]
_top_right_corner(grid::AbstractRegularMesh2D, i_cell) = cell_center(grid, i_cell) .+ @SVector [dx(grid)/2, dy(grid)/2]
 
using PolygonArea
_cell_as_polygon(grid::AbstractRegularMesh2D, i_cell) = rectangle(_bottom_left_corner(grid, i_cell)...,
                                                                 _top_right_corner(grid, i_cell)...)

function integrate(poly::PolygonArea.ConvexPolygon, grid::AbstractRegularMesh2D)
    α = zeros(SVector{1, Float64}, nb_cells(grid))
    for i_cell in 1:nb_cells(grid)
        α[i_cell] = @SVector [area(poly ∩ _cell_as_polygon(grid, i_cell))/cell_volume(grid, i_cell)]
    end
    return α
end

function rotation_matrix(grid::AbstractRegularMesh2D, i_face)
    if _is_horizontal(i_face)
        return SMatrix{2, 2, Float64}(0.0, 1.0, -1.0, 0.0)
    else
        return SMatrix{2, 2, Float64}(1.0, 0.0, 0.0, 1.0)
    end
end

# Indices of the cells in the 3x3 stencil around i_cell
function stencil(grid::AbstractRegularMesh2D, i_cell)
	if i_cell % grid.nx == 0  # Last cell at the end of a row
        if grid isa PeriodicRegularMesh2D
            right_cell = i_cell + 1 - grid.nx
        else
            right_cell = i_cell  # "Neumann" stencil
        end
	else  # General case
		right_cell = i_cell + 1
	end

	if i_cell % grid.nx == 1  # First cell at the beggining of a row
        if grid isa PeriodicRegularMesh2D
            left_cell = i_cell - 1 + grid.nx
        else
            left_cell = i_cell
        end
	else
		left_cell = i_cell - 1
	end

	stencil = @SMatrix [left_cell-grid.nx   left_cell    left_cell+grid.nx;
						i_cell-grid.nx      i_cell       i_cell+grid.nx;
                        right_cell-grid.nx  right_cell   right_cell+grid.nx]

    if grid isa PeriodicRegularMesh2D
	    stencil = (s -> mod(s, Base.OneTo(nb_cells(grid)))).(stencil)
    else
        stencil = (s -> s < 1 ? s + grid.nx : (s > nb_cells(grid) ? s - grid.nx : s)).(stencil)
    end

	return OffsetArray(stencil, -1:1, -1:1)
end

function _rotr90(st)
    return OffsetArray(
                       @SMatrix [st[1, -1]  st[0, -1]  st[-1, -1];
                                 st[1, 0]   st[0, 0]   st[-1, 0];
                                 st[1, 1]   st[0, 1]   st[-1, 1]]
                       , -1:1, -1:1)
end

function _rotl90(st)
    return OffsetArray(
                       @SMatrix [st[-1, 1]  st[0, 1]  st[1, 1];
                                 st[-1, 0]  st[0, 0]  st[1, 0];
                                 st[-1, -1] st[0, -1] st[1, -1]]
                       , -1:1, -1:1)
end

function _rot180(st)
    return OffsetArray(
                       @SMatrix [st[1, 1]   st[1, 0]   st[1, -1];
                                 st[0, 1]   st[0, 0]   st[0, -1];
                                 st[-1, 1]  st[-1, 0]  st[-1, -1]]
                       , -1:1, -1:1)
end

function _transpose(st)
    return OffsetArray(
                       @SMatrix [st[-1, -1]  st[0, -1]  st[1, -1];
                                 st[-1, 0]   st[0, 0]   st[1, 0];
                                 st[-1, 1]   st[0, 1]   st[1, 1]]
                       , -1:1, -1:1)

end

function oriented_stencil(mesh::AbstractRegularMesh2D, i_cell, i_face)
    st = stencil(mesh, i_cell)
    if _is_horizontal(i_face) && i_cell == cells_next_to_inner_face(mesh, i_face)[1]
        st = _rotr90(st)
    elseif _is_horizontal(i_face) && i_cell == cells_next_to_inner_face(mesh, i_face)[2]
        st = _rotl90(st)
    elseif !_is_horizontal(i_face) && i_cell == cells_next_to_inner_face(mesh, i_face)[2]
        st = _rot180(st) 
    end
    return st
end

function central_differences_gradient(grid::PeriodicRegularMesh2D, w, i_cell)
	st = stencil(grid, i_cell)
	dwdx = (w[st[1, 0]] - w[st[-1, 0]])/(2dx(grid))
	dwdy = (w[st[0, 1]] - w[st[0, -1]])/(2dy(grid))
	return SVector{2, Float64}(dwdx, dwdy)
end

function youngs_gradient(grid::PeriodicRegularMesh2D, w, i_cell)
	st = stencil(grid, i_cell)
	dwdx = (w[st[1, 1]] + 2w[st[1, 0]] + w[st[1, -1]] - w[st[-1, 1]] - 2w[st[-1, 0]] - w[st[-1, -1]])/(8dx(grid))
	dwdy = (w[st[1, 1]] + 2w[st[0, 1]] + w[st[-1, 1]] - w[st[1, -1]] - 2w[st[0, -1]] - w[st[-1, -1]])/(8dy(grid))
	return SVector{2, Float64}(dwdx, dwdy)
end

function least_square_gradient(grid::PeriodicRegularMesh2D, w, i_cell)
	st = stencil(grid, i_cell)
    A = @SMatrix zeros(2, 2)
    b = @SVector zeros(2)
	for i in -1:1, j in -1:1
		Δx = i*dx(grid)
		Δy = j*dy(grid)
		Δu = w[st[i, j]] - w[st[0, 0]]
		A = A + SMatrix{2, 2, Float64}(Δx^2, Δx*Δy, Δx*Δy, Δy^2)
		b = b + SVector{2, Float64}(Δu*Δx, Δu*Δy)
	end

    if A[2, 2] == 0.0
        #= @warning "2D mesh that is actually 1D" =#
        return [b[1] / A[1, 1], 0.0]
    else
        return A \ b
    end
end

