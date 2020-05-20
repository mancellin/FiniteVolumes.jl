import Base.getindex
import Base.size
import Base.rotr90
import Base.rotl90
import Base.rot180
import Base.transpose

struct Stencil{NY, NX, T}
    data::SMatrix{NY, NX, T}
end

Stencil(s::AbstractMatrix) = Stencil{size(s)..., eltype(s)}(SMatrix{size(s)..., eltype(s)}(s...))


offset_i(::Stencil{NY, NX, T}) where {NX, NY, T} = (NY - 1) ÷ 2 + 1
offset_j(::Stencil{NY, NX, T}) where {NX, NY, T} = (NX - 1) ÷ 2 + 1

getindex(s::Stencil, i::Int, j::Int) = getindex(s.data, i + offset_i(s), j + offset_j(s))
size(s::Stencil) = size(s.data)

function rotr90(st::Stencil{3, 3, T}) where T
    return Stencil{3, 3, T}(
                            @SMatrix [st[1, -1]  st[0, -1]  st[-1, -1];
                                      st[1, 0]   st[0, 0]   st[-1, 0];
                                      st[1, 1]   st[0, 1]   st[-1, 1]]
                           )
end

function rotl90(st::Stencil{3, 3, T}) where T
    return Stencil{3, 3, T}(
                            @SMatrix [st[-1, 1]  st[0, 1]  st[1, 1];
                                      st[-1, 0]  st[0, 0]  st[1, 0];
                                      st[-1, -1] st[0, -1] st[1, -1]]
                           )
end

function rot180(st::Stencil{3, 3, T}) where T
    return Stencil{3, 3, T}(
                            @SMatrix [st[1, 1]   st[1, 0]   st[1, -1];
                                      st[0, 1]   st[0, 0]   st[0, -1];
                                      st[-1, 1]  st[-1, 0]  st[-1, -1]]
                           )
end

function transpose(st::Stencil{3, 3, T}) where T
    return Stencil{3, 3, T}(
                       @SMatrix [st[-1, -1]  st[0, -1]  st[1, -1];
                                 st[-1, 0]   st[0, 0]   st[1, 0];
                                 st[-1, 1]   st[0, 1]   st[1, 1]]
                      )
end

