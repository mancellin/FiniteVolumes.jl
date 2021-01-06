
abstract type FiniteVolumeMesh end

cell_centers(m::FiniteVolumeMesh) = [cell_center(m, i) for i in all_cells(m)]

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

all_cells(grid::RegularMesh1D) = 1:nb_cells(grid)
cell_center(grid::RegularMesh1D, i_cell) = SVector{1, Float64}(dx(grid)*(Float64(i_cell) - 0.5))
face_center(grid::RegularMesh1D, i_face) = SVector{1, Float64}(dx(grid)*(Float64(i_face) - 1.0))

@inline inner_faces(grid::RegularMesh1D) = Vector(2:(nb_faces(grid)-1))
@inline cells_next_to_inner_face(grid::RegularMesh1D, i_face)::Tuple{Int64, Int64} = (i_face-1, i_face)

@inline boundary_faces(grid::RegularMesh1D) = [1, nb_faces(grid)]
@inline cell_next_to_boundary_face(grid::RegularMesh1D, i_face)::Int64 = i_face == 1 ? 1 : nb_cells(grid)

@inline face_area(grid::RegularMesh1D, i_face)::Float64 = 1.0
@inline cell_volume(grid::RegularMesh1D, i_cell)::Float64 = dx(grid)

@inline rotation_matrix(grid::RegularMesh1D, i_face) = SMatrix{1, 1, Float64}(i_face == 1 ? -1.0 : 1.0)

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
all_cells(grid::AbstractRegularMesh2D) = 1:nb_cells(grid)

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

function face_center(grid::AbstractRegularMesh2D, i_face)
    if grid isa PeriodicRegularMesh2D || _is_inner_face(grid, i_face)
        i_cell, _ = cells_next_to_inner_face(grid, i_face)
    else
        i_cell = cell_next_to_boundary_face(grid, i_face)
    end
    return (cell_center(grid, i_cell) .+ 
            rotation_matrix(grid, i_face)[:, 1] .* @SVector [dx(grid)/2, dy(grid)/2])
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

function rotation_matrix(mesh::RegularMesh2D, i_face)
    if _is_inner_face(mesh, i_face) || _is_on_top_edge(mesh, i_face) || _is_on_right_edge(mesh, i_face)
        if _is_horizontal(i_face)
            return SMatrix{2, 2, Float64}(0.0, 1.0, -1.0, 0.0)
        else
            return SMatrix{2, 2, Float64}(1.0, 0.0, 0.0, 1.0)
        end
    elseif _is_on_bottom_edge(mesh, i_face)
        return SMatrix{2, 2, Float64}(0.0, -1.0, 1.0, 0.0)
    elseif _is_on_left_edge(mesh, i_face)
        return SMatrix{2, 2, Float64}(-1.0, 0.0, 0.0, -1.0)
    else
        error("This should not happen")
    end
end

function rotation_matrix(grid::PeriodicRegularMesh2D, i_face)
    if _is_horizontal(i_face)
        return SMatrix{2, 2, Float64}(0.0, 1.0, -1.0, 0.0)
    else
        return SMatrix{2, 2, Float64}(1.0, 0.0, 0.0, 1.0)
    end
end

function cell_corners(mesh::AbstractRegularMesh2D, i_cell)
    c = cell_center(mesh, i_cell)
    return (
     bottom_left= SVector(c[1] - dx(mesh)/2, c[2] - dy(mesh)/2),
     top_left=    SVector(c[1] - dx(mesh)/2, c[2] + dy(mesh)/2),
     bottom_right=SVector(c[1] + dx(mesh)/2, c[2] - dy(mesh)/2),
     top_right=   SVector(c[1] + dx(mesh)/2, c[2] + dy(mesh)/2),
    )
end


