module CartesianMeshes

using StaticArrays

export AbstractCartesianMesh, CartesianMesh, PeriodicCartesianMesh
export nb_dims, nb_cells, all_cells
export nb_inner_faces, inner_faces
export nb_boundary_faces, boundary_faces
export cell_centers, cells_centers, cell_center, face_center
export cells_next_to_inner_face, cell_next_to_boundary_face
export cell_volume, face_area, normal_vector

# Tools
struct Half{N} <: Real
    n::N
end
import Base.Int, Base.convert, Base.+, Base.-, Base.<, Base.==, Base.print
Base.Int(h::Half{Int}) = Base.div(h.n, 2)
Base.convert(::Type{Int}, h::Half{Int}) = Int(h)
Base.convert(::Type{<:Number}, h::Half) = h.n/2
Base.:+(a::Half, b::Half) = Half(a.n+b.n)
Base.:-(a::Half) = Half(-a.n)
Base.:-(a::Half, b::Half) = Half(a.n-b.n)
Base.:(==)(a::Half, b::Half) = a.n == b.n
Base.:<(a::Half, b::Half) = a.n < b.n
is_int(h::Half{Int}) = mod(h.n, 2) == 0
Base.print(io::IO, h::Half) = print(io, "$(h.n)/2")



cell_centers(mesh) = map(i -> cell_center(mesh, i), all_cells(mesh))
cells_centers(mesh) = cell_centers(mesh)

const FaceIndex{N} = NTuple{N, Half{Int}}
const CellIndex{N} = CartesianIndex{N}

#################################################################
abstract type AbstractCartesianMesh{D, L} end
#################################################################

#########################################################
struct CartesianMesh{D, L} <: AbstractCartesianMesh{D, L}
    x_min::SVector{D, L}
    x_max::SVector{D, L}
    nb_cells::SVector{D, Int64}
end
CartesianMesh(x_min::Number, x_max::Number, nb_cells::Int) = CartesianMesh{1, typeof(x_min)}(SVector(x_min), SVector(x_max), (nb_cells,))
CartesianMesh(nb_cells::Int) = CartesianMesh{1, Float64}(SVector(0.0), SVector(1.0), (nb_cells,))
CartesianMesh(nx::Int, ny::Int) = CartesianMesh{2, Float64}(SVector(0.0, 0.0), SVector(1.0, 1.0), (nx, ny))
CartesianMesh(nx::Int, ny::Int, nz::Int) = CartesianMesh{3, Float64}(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0), (nx, ny, nz))

CartesianMesh(x_min::NTuple{N}, x_max::NTuple{N}, nb_cells::NTuple{N, Int64}) where N = CartesianMesh{N, eltype(x_min)}(SVector(x_min...), SVector(x_max...), SVector(nb_cells...))

#################################################################
struct PeriodicCartesianMesh{D, L} <: AbstractCartesianMesh{D, L}
    x_min::SVector{D, L}
    x_max::SVector{D, L}
    nb_cells::SVector{D, Int64}
end
PeriodicCartesianMesh(x_min::Number, x_max::Number, nb_cells::Int) = PeriodicCartesianMesh{1, typeof(x_min)}(SVector(x_min), SVector(x_max), (nb_cells,))
PeriodicCartesianMesh(nb_cells::Int) = PeriodicCartesianMesh{1, Float64}(SVector(0.0), SVector(1.0), (nb_cells,))
PeriodicCartesianMesh(nx::Int, ny::Int) = PeriodicCartesianMesh{2, Float64}(SVector(0.0, 0.0), SVector(1.0, 1.0), (nx, ny))
PeriodicCartesianMesh(nx::Int, ny::Int, nz::Int) = PeriodicCartesianMesh{3, Float64}(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0), (nx, ny, nz))

PeriodicCartesianMesh(x_min::NTuple{N}, x_max::NTuple{N}, nb_cells::NTuple{N, Int64}) where N = PeriodicCartesianMesh{N, eltype(x_min)}(SVector(x_min...), SVector(x_max...), SVector(nb_cells...))

nb_dims(mesh::AbstractCartesianMesh{D}) where D = D

dx(mesh::AbstractCartesianMesh) = @. (mesh.x_max - mesh.x_min)/mesh.nb_cells

nb_cells(mesh::AbstractCartesianMesh) = prod(mesh.nb_cells)
all_cells(mesh::AbstractCartesianMesh) = CartesianIndices(Tuple(mesh.nb_cells))

# INNER FACES
nb_inner_faces(mesh::CartesianMesh{1}) = mesh.nb_cells[1] - 1
inner_faces(mesh::CartesianMesh{1}) = ((Half(n),) for n in 3:2:2*mesh.nb_cells[1])
nb_boundary_faces(mesh::CartesianMesh{1}) = 2
boundary_faces(mesh::CartesianMesh{1}) = ((Half(1),), (Half(2*mesh.nb_cells[1]+1),))

nb_inner_faces(mesh::PeriodicCartesianMesh{1}) = mesh.nb_cells[1]
inner_faces(mesh::PeriodicCartesianMesh{1}) = ((Half(n),) for n in 1:2:2*mesh.nb_cells[1])
nb_boundary_faces(mesh::PeriodicCartesianMesh{1}) = 0
boundary_faces(mesh::PeriodicCartesianMesh{1}) = Tuple([]) 

const _left = (-Half(1), Half(0), Half(0))
const _right = (Half(1), Half(0), Half(0))
const _down = (Half(0), -Half(1), Half(0))
const _up = (Half(0), Half(1), Half(0))
const _back = (Half(0), Half(0), -Half(1))
const _front = (Half(0), Half(0), Half(1))
_bottom_cells(mesh::CartesianMesh{2}) = ((n, m) = mesh.nb_cells; CartesianIndices((1:n, 1:1)))
_top_cells(mesh::CartesianMesh{2}) = ((n, m) = mesh.nb_cells; CartesianIndices((1:n, m:m)))
_left_cells(mesh::CartesianMesh{2}) = ((n, m) = mesh.nb_cells; CartesianIndices((1:1, 1:m)))
_right_cells(mesh::CartesianMesh{2}) = ((n, m) = mesh.nb_cells; CartesianIndices((n:n, 1:m)))

_face(dir, cell::CellIndex{N}) where N = Half.(2 .* Tuple(cell)) .+ dir[1:N]

nb_inner_faces(mesh::CartesianMesh{2}) = 2 * mesh.nb_cells[1] * mesh.nb_cells[2] - mesh.nb_cells[1] - mesh.nb_cells[2]
inner_faces(mesh::CartesianMesh{2}) = ((n, m) = mesh.nb_cells; (_face(dir, cell) for (dir, cells) in ((_right, CartesianIndices((1:(n-1), 1:m))), (_up, CartesianIndices((1:n, 1:(m-1))))) for cell in cells))
nb_boundary_faces(mesh::CartesianMesh{2}) = 2 * (mesh.nb_cells[1] + mesh.nb_cells[2])
boundary_faces(mesh::CartesianMesh{2}) = (_face(dir, cell) for (dir, cells) in ((_down, _bottom_cells(mesh)), (_right, _right_cells(mesh)), (_up, _top_cells(mesh)), (_left, _left_cells(mesh))) for cell in cells)

nb_inner_faces(mesh::PeriodicCartesianMesh) = 2 * prod(mesh.nb_cells)
inner_faces(mesh::PeriodicCartesianMesh{N}) where N = (_face(dir, cell) for cell in CartesianIndices(Tuple(mesh.nb_cells)) for dir in (_left, _down, _back)[1:N])
nb_boundary_faces(mesh::PeriodicCartesianMesh) = 0
boundary_faces(mesh::PeriodicCartesianMesh) = Tuple([])

# CELLS NEXT TO FACE
_direction(i_face::FaceIndex{1}) = 1
_direction(i_face::FaceIndex{2}) = !is_int(i_face[1]) ? 1 : 2  # 1 = horizontal, 2 = vertical
_direction(i_face::FaceIndex{3}) = !is_int(i_face[1]) ? 1 : !is_int(i_face[2]) ? 2 : 3

_dir_step(i_face::FaceIndex{1}) = (1,)
_dir_step(i_face::FaceIndex{2}) = _direction(i_face) == 1 ? (1, 0) : (0, 1)
_dir_step(i_face::FaceIndex{3}) = _direction(i_face) == 1 ? (1, 0, 0) : _direction(i_face) == 2 ? (0, 1, 0) : (0, 0, 1)

_cells_next(i_face::FaceIndex{N}) where N = (Int.(i_face), Int.(i_face) .+ _dir_step(i_face))
_cells_next(i_face::Half{Int}) = _cells_next((i_face,))

cells_next_to_inner_face(::CartesianMesh, i_face) = CellIndex.(_cells_next(i_face))
function cells_next_to_inner_face(mesh::PeriodicCartesianMesh{N}, i_face::FaceIndex{N}) where N
    if Half(1) in i_face
        i_cell_1, i_cell_2 = _cells_next(i_face)
        return (CellIndex(mod1.(i_cell_1, Tuple(mesh.nb_cells))),
                CellIndex(mod1.(i_cell_2, Tuple(mesh.nb_cells))))
    else
        return CellIndex.(_cells_next(i_face))
    end
end

cell_next_to_boundary_face(mesh::CartesianMesh{1}, i_face) = i_face == (Half(1),) ? CellIndex(1) : CellIndex(mesh.nb_cells[1])

function cell_next_to_boundary_face(mesh::CartesianMesh{2}, i_face)
    if i_face[1] == Half(1)  # left boundary
        return CellIndex(Int(i_face[1]) + 1, Int(i_face[2]))
    elseif i_face[2] == Half(1)  # bottom boundary
        return CellIndex(Int(i_face[1]), Int(i_face[2]) + 1)
    elseif _direction(i_face) == 1  # right boundary
        return CellIndex(Int(i_face[1]), Int(i_face[2]))
    elseif _direction(i_face) == 2  # top boundary
        return CellIndex(Int(i_face[1]), Int(i_face[2]))
    else
        error()
    end
end

# GEOMETRY
cell_center(mesh::AbstractCartesianMesh{1}, i_cell) = (i_cell[1] - 0.5) * dx(mesh)[1]
cell_center(mesh::AbstractCartesianMesh{N}, i_cell) where N = (SVector{N}(Tuple(i_cell)...) .- 0.5) .* dx(mesh)
face_center(mesh::AbstractCartesianMesh{1}, i_face) = (i_face[1] - 0.5) * dx(mesh)[1]
face_center(mesh::AbstractCartesianMesh{N}, i_face) where N = (SVector{N, Float64}(Tuple(i_face)...) .- 0.5) .* dx(mesh)


cell_volume(mesh::AbstractCartesianMesh, i_cell) = prod(dx(mesh))


face_area(mesh::AbstractCartesianMesh{1, T}, i_face) where T = one(T)
face_area(mesh::AbstractCartesianMesh{2}, i_face) = _direction(i_face) == 1 ? dx(mesh)[2] : dx(mesh)[1]
function face_area(mesh::AbstractCartesianMesh{3}, i_face) 
    Δx, Δy, Δz = dx(mesh)
    if _direction(i_face) == 1; return Δy*Δz
    elseif _direction(i_face) == 2; return Δx*Δz
    elseif _direction(i_face) == 3; return Δx*Δy
    end
end

normal_vector(mesh::CartesianMesh{1, T}, i_face) where T = i_face == (Half(1),) ? -one(T) : one(T)
normal_vector(mesh::PeriodicCartesianMesh{1, T}, i_face) where T = one(T)


_normal(T, i_face) = map(x -> x == 1 ? one(T) : zero(T)/oneunit(T), _dir_step(i_face))
# Division by oneunit is required for compatibility with Unitful
#
function normal_vector(::PeriodicCartesianMesh{N, T}, i_face) where {N, T}
    SVector(_normal(T, i_face)...)
end
function normal_vector(mesh::AbstractCartesianMesh{N, T}, i_face) where {N, T}
    n = SVector(_normal(T, i_face)...)
    if Half(1) in i_face
        return -n
    else
        return n
    end
end

include("./plot_recipes.jl")

end # module
