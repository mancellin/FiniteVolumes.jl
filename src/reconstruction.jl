identity(x) = x

function upwind_cell(grid, model::ScalarLinearAdvection, w, wsupp, i_face)
    local_model = rotate_model(model, rotation_matrix(grid, i_face))
    local_velocity = local_model.velocity[1]
    i_cell_1, i_cell_2 = cells_next_to_inner_face(grid, i_face)
    up_cell = local_velocity > 0.0 ? i_cell_1 : i_cell_2
    return local_velocity, up_cell
end

function upwind_stencil(grid, model::ScalarLinearAdvection, w, wsupp, i_face)
    local_velocity, up_cell = upwind_cell(grid, model, w, wsupp, i_face)
    st = oriented_stencil(grid, up_cell, i_face)
    if ndims(st) == 1
        wst = OffsetArray(@SVector [w[st[-1]], w[st[0]], w[st[1]]]
                          , -1:1)
    elseif ndims(st) == 2
        wst = OffsetArray(
                          @SMatrix [w[st[-1, -1]] w[st[-1, 0]] w[st[-1, 1]];
                                    w[st[0, -1]]  w[st[0, 0]]  w[st[0, 1]];
                                    w[st[1, -1]]  w[st[1, 0]]  w[st[1, 1]]]
                          , -1:1, -1:1)
    end
    return local_velocity, wst
end


###########
#  Muscl  #
###########

Base.@kwdef struct Muscl{L, R} <: NumericalFlux
    limiter::L
    renormalize::R = identity
end

minmod      = (a, b) -> a*b <= 0 ? 0.0 : (a >= 0 ? min(a, b) : max(a, b))
superbee    = (a, b) -> a*b <= 0 ? 0.0 : (a >= 0 ? max(min(2*a, b), min(a, 2*b)) : min(max(2*a, b), max(a, 2*b)))
ultrabee(β) = (a, b) -> a*b <= 0 ? 0.0 : (a >= 0 ? 2*max(0, min((1/β-1)*a, b)) : -ultrabee(β)(-a, -b))


function (s::Muscl)(grid, model::ScalarLinearAdvection, w, wsupp, i_face)
    if grid isa RegularMesh1D
        v, wst = upwind_stencil(grid, model, w, wsupp, i_face)
    else
        v, wst2d = upwind_stencil(grid, model, w, wsupp, i_face)
        wst = OffsetArray(SVector(wst2d[-1, 0], wst2d[0, 0], wst2d[1, 0]), -1:1)
    end
    grad_w::eltype(w) = s.limiter.(wst[0] - wst[-1], wst[1] - wst[0])
    re_w::eltype(w) = wst[0] .+ 0.5 * grad_w
    rere_w::eltype(w) = s.renormalize(re_w)
    return eltype(w)(v * rere_w)
end

#########
#  VOF  #
#########

Base.@kwdef struct VOF{M} <: NumericalFlux
    method::M
    β::Float64
end

function (s::VOF)(grid, model::ScalarLinearAdvection, w, wsupp, i_face)
    v, wst = FiniteVolumes.upwind_stencil(grid, model, w, wsupp, i_face)
    α_flux = s.method(wst, s.β)
    return eltype(w)(v * α_flux)
end


########################
#  LagoutiereDownwind  #
########################

Base.@kwdef struct LagoutiereDownwind <: NumericalFlux
    β::Float64
end

function stability_range(α, β)
    maxi = max(α[-1], α[0])
    mini = min(α[-1], α[0])
    bornesup = min((α[0] - mini)/β + mini, max(α[0], α[1]))
    borneinf = max((α[0] - maxi)/β + maxi, min(α[0], α[1]))
    return (borneinf, bornesup)
end

cut_in_range(inf, sup, x) = min(sup, max(inf, x))

function (s::LagoutiereDownwind)(grid, model::ScalarLinearAdvection{1, T, D}, w, wsupp, i_face) where {T, D}
    if grid isa RegularMesh1D
        v, wst = upwind_stencil(grid, model, w, wsupp, i_face)
    else
        v, wst2d = upwind_stencil(grid, model, w, wsupp, i_face)
        wst = OffsetArray(SVector(wst2d[-1, 0], wst2d[0, 0], wst2d[1, 0]), -1:1)
    end
    borneinf, bornesup = stability_range(wst, s.β)
    α_flux = cut_in_range(borneinf, bornesup, wst[1])
    return eltype(w)(v * α_flux)
end

function (s::LagoutiereDownwind)(grid, model::ScalarLinearAdvection{N, T, D}, w, wsupp, i_face) where {N, T, D}
    if grid isa RegularMesh1D
        v, wst = upwind_stencil(grid, model, w, wsupp, i_face)
    else
        v, wst2d = upwind_stencil(grid, model, w, wsupp, i_face)
        wst = OffsetArray(SVector(wst2d[0, -1], wst2d[0, 0], wst2d[0, 1]), -1:1)
    end

    bornesup = @MVector zeros(nb_vars(model))
    borneinf = @MVector zeros(nb_vars(model))
    for i in 1:nb_vars(model)
        wi = (x -> x[i]).(wst)
        borneinf[i], bornesup[i] = stability_range(wi, s.β)
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

    return eltype(w)(v*α_flux)
end


############
#  Either  #
############

all_cells(args...) = true
no_cell(args...) = false

struct Either{C, F1, F2} <: NumericalFlux
    condition::C
    flux_true::F1
    flux_false::F2
end

function (s::Either)(args...)
    if s.condition(args...)
        s.flux_true(args...)
    else
        s.flux_false(args...)
    end
end

