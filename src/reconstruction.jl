################################################################################
#                            Scalar reconstruction                             #
################################################################################

function upwind_cell(model::ScalarLinearAdvection, mesh, i_face)
    local_model = rotate_model(model, rotation_matrix(mesh, i_face), face_center(mesh, i_face))
    local_velocity = local_model.velocity[1]
    i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
    up_cell = local_velocity > 0.0 ? i_cell_1 : i_cell_2
    return local_velocity, up_cell
end

function upwind_stencil(model, mesh::RegularMesh1D, w, i_face, N)
    u, i_cell = upwind_cell(model, mesh, i_face)
    st = Stencil{N, 1}(mesh, i_cell, i_face)
    return u, w[st], dx(mesh)
end

function upwind_stencil(model, mesh::AbstractRegularMesh2D, w, i_face, N, M=1)
    u, i_cell = upwind_cell(model, mesh, i_face)
    st = Stencil{N, M}(mesh, i_cell, i_face)
    Δx = _is_horizontal(i_face) ? dy(mesh) : dx(mesh)
    return u, w[st], Δx
end

###########
#  Muscl  #
###########

identity(x) = x

Base.@kwdef struct Muscl{L, R} <: NumericalFlux
    limiter::L
    renormalize::R = identity
end

minmod(a, b, β) = a*b <= 0 ? 0.0 : (a >= 0 ? min(a, b) : max(a, b))
superbee(a, b, β) = a*b <= 0 ? 0.0 : (a >= 0 ? max(min(2*a, b), min(a, 2*b)) : min(max(2*a, b), max(a, 2*b)))
ultrabee(a, b, β) = a*b <= 0 ? 0.0 : (a >= 0 ? 2*max(0.0, min((1/β-1)*a, b)) : -ultrabee(-a, -b, β))

function (s::Muscl)(model::ScalarLinearAdvection, mesh, w, i_face; dt=0.0)
    u, wst, Δx = upwind_stencil(model, mesh, w, i_face, 3)
    if isnothing(dt)
        β = nothing
    else
        β = dt*abs(u)/Δx
    end
    grad_w::eltype(w) = s.limiter.(wst[0] - wst[-1], wst[1] - wst[0], β)
    re_w::eltype(w) = wst[0] .+ 0.5 * grad_w
    rere_w::eltype(w) = s.renormalize(re_w)
    return eltype(w)(u * rere_w)
end

Base.print(io::IO, m::Muscl) = print(io, "Muscl($(m.limiter))")

#########
#  VOF  #
#########


struct VOF{NX, NY, M} <: NumericalFlux
    method::M
end

VOF(; method) = VOF(method)
VOF(method) = VOF{3, 3}(method)
VOF{NX, NY}(method) where {NX, NY} = VOF{NX, NY, typeof(method)}(method)

function (s::VOF{NX, NY})(model::ScalarLinearAdvection, mesh, w, i_face; dt=0.0) where {NX, NY}
    stencil_radiuses = ((NX - 1) ÷ 2, (NY - 1) ÷ 2)
    u, wst, Δx = upwind_stencil(model, mesh, w, i_face, NX, NY)
    if abs(u) < 1e-10
        return zero(eltype(w))
    else
        α_flux = s.method(wst, dt*abs(u)/Δx)
        return eltype(w)(u * α_flux)
    end
end

Base.print(io::IO, s::VOF{NX, NY}) where {NX, NY} = print(io, "VOF{$NX, $NY}($(s.method))")


########################
#  LagoutiereDownwind  #
########################

struct LagoutiereDownwind <: NumericalFlux end

function stability_range(α, β)
    maxi = max(α[-1], α[0])
    mini = min(α[-1], α[0])
    bornesup = min((α[0] - mini)/β + mini, max(α[0], α[1]))
    borneinf = max((α[0] - maxi)/β + maxi, min(α[0], α[1]))
    return (borneinf, bornesup)
end

cut_in_range(inf, sup, x) = min(sup, max(inf, x))

function (s::LagoutiereDownwind)(model::ScalarLinearAdvection{1, T, D}, mesh, w, i_face; dt=0.0) where {T, D}
    u, wst, Δx = upwind_stencil(model, mesh, w, i_face, 3)
    if abs(u) < 1e-10
        return zero(eltype(w))
    else
        borneinf, bornesup = stability_range(wst, dt*abs(u)/Δx)
        α_flux = cut_in_range(borneinf, bornesup, wst[1])
        return eltype(w)(u * α_flux)
    end
end

function (s::LagoutiereDownwind)(model::ScalarLinearAdvection{N, T, D}, mesh, w, i_face; dt=0.0) where {N, T, D}
    u, wst, Δx = upwind_stencil(model, mesh, w, i_face, 3)
    if abs(u) < 1e-10
        return zero(eltype(w))
    else
        bornesup = @MVector zeros(nb_vars(model))
        borneinf = @MVector zeros(nb_vars(model))
        for i in 1:nb_vars(model)
            wi = map(x -> x[i], wst)
            borneinf[i], bornesup[i] = stability_range(wi, dt*abs(u)/Δx)
        end

        α_flux = @MVector zeros(nb_vars(model))
        for i in 1:nb_vars(model)
            sumα = i > 1 ? sum(α_flux[j] for j in 1:(i-1)) : 0.0
            sumsup = i < nb_vars(model) ? sum(bornesup[j] for j in (i+1):nb_vars(model)) : 0.0
            suminf = i < nb_vars(model) ? sum(borneinf[j] for j in (i+1):nb_vars(model)) : 0.0
            updated_borneinf = max(borneinf[i], 1.0 - sumα - sumsup)
            updated_bornesup = min(bornesup[i], 1.0 - sumα - suminf)
            α_flux[i] = min(updated_bornesup, max(updated_borneinf, wst[1][i]))
        end

        return eltype(w)(u*α_flux)
    end
end


############
#  Hybrid  #
############

all_cells(args...) = true
no_cell(args...) = false

struct Hybrid{C, F1, F2} <: NumericalFlux
    condition::C
    flux_true::F1
    flux_false::F2
end

function (s::Hybrid)(args...; kwargs...)
    if s.condition(args...; kwargs...)
        s.flux_true(args...; kwargs...)
    else
        s.flux_false(args...; kwargs...)
    end
end

Base.print(io::IO, c::Hybrid) = print(io, "Hybrid($(c.condition), $(c.flux_true), $(c.flux_false))")

# Example of conditions for scalar advection
struct InMixedCells{T}
    threshold::T
end

function (c::InMixedCells)(model::ScalarLinearAdvection{1}, mesh, w, i_face; dt=nothing)
    c.threshold <= w[FiniteVolumes.upwind_cell(model, mesh, i_face)[2]][1] <= (1.0-c.threshold)
end

function (c::InMixedCells)(model::ScalarLinearAdvection, mesh, w, i_face; dt=nothing)
    any(c.threshold .<= w[FiniteVolumes.upwind_cell(model, mesh, i_face)[2]] .<= (1.0-c.threshold))
end

Base.print(io::IO, c::InMixedCells) = print(io, "InMixedCells($(c.threshold))")
