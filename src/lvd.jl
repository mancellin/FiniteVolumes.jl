
function faces_after_cell(mesh, cell)
    bc(i) = (Half(mod1(i[1].n, 2*mesh.nb_cells[1])), Half(mod1(i[2].n, 2*mesh.nb_cells[2])))
    faces =  collect(inner_faces(mesh))
    face_1 = bc(Half.(2 .* Tuple(cell)) .+ (Half(1), Half(0)))
    i_face_1 = findfirst(==(face_1), faces)
    face_2 = bc(Half.(2 .* Tuple(cell)) .+ (Half(0), Half(1)))
    i_face_2 = findfirst(==(face_2), faces)
    return i_face_1, i_face_2
end

function cells_before_cell(mesh, cell)
    bc(i) = (mod1(i[1], mesh.nb_cells[1]), mod1(i[2], mesh.nb_cells[2]))
    return CartesianIndex(bc(Tuple(cell) .- (1, 0))), CartesianIndex(bc(Tuple(cell) .- (0, 1)))
end

function positivity_flagged_cells(flux::LinearAdvectionFlux, mesh::PeriodicCartesianMesh{2}, w, Φ, dt)
    flagged_cells = zeros(Bool, axes(w))
    a, b = flux.velocity
    p, q = a/(a+b), b/(a+b)
    β = dt * (a+b) * face_area(mesh, (Half(1), Half(1))) / cell_volume(mesh, CartesianIndex(1, 1))
    for cell in all_cells(mesh)
        i_face_1, i_face_2 = faces_after_cell(mesh, cell)
        mϕ = Φ[i_face_1] + Φ[i_face_2]
        flagged_cells[cell] = !(1 - (1 - w[cell])/β <= mϕ <= w[cell]/β)
    end
    return flagged_cells
end

function lvd_flagged_cells(flux::LinearAdvectionFlux, mesh::PeriodicCartesianMesh{2}, w, Φ, dt)
    flagged_cells = zeros(Bool, axes(w))
    a, b = flux.velocity
    p, q = a/(a+b), b/(a+b)
    for cell in all_cells(mesh)
        β = dt * (a + b) * face_area(mesh, (Half(1), Half(1))) / cell_volume(mesh, cell)
        i_face_1, i_face_2 = faces_after_cell(mesh, cell)
        mϕ = Φ[i_face_1] + Φ[i_face_2]
        cell_i, cell_j = cells_before_cell(mesh, cell)
        lb = (
              p*(min(w[cell] - w[cell_i], 0)/β + max(w[cell], w[cell_i]))
              + q*(min(w[cell] - w[cell_j], 0)/β + max(w[cell], w[cell_j]))
             )
        ub = (
              p*(max(w[cell] - w[cell_i], 0)/β + min(w[cell], w[cell_i]))
              + q*(max(w[cell] - w[cell_j], 0)/β + min(w[cell], w[cell_j]))
             )
        flagged_cells[cell] = !(lb <= mϕ <= ub)
    end
    return flagged_cells
end

function lvd_correction(flux::LinearAdvectionFlux, mesh::PeriodicCartesianMesh{2}, w, Φ, dt)
    Ψ = copy(Φ)
    a, b = flux.velocity
    β = dt * (a + b) * face_area(mesh, (Half(1), Half(1))) / cell_volume(mesh, CartesianIndex(1, 1))
    for cell in all_cells(mesh)
        i_face_1, i_face_2 = faces_after_cell(mesh, cell)
        cell_i, cell_j = cells_before_cell(mesh, cell)
        lb = (
              a*(min(w[cell] - w[cell_i], 0)/β + max(w[cell], w[cell_i]))
              + b*(min(w[cell] - w[cell_j], 0)/β + max(w[cell], w[cell_j]))
             )
        ub = (
              a*(max(w[cell] - w[cell_i], 0)/β + min(w[cell], w[cell_i]))
              + b*(max(w[cell] - w[cell_j], 0)/β + min(w[cell], w[cell_j]))
             )
        if Φ[i_face_1] + Φ[i_face_2] > ub
            Ψ[i_face_1] = Φ[i_face_1] - (Φ[i_face_1] + Φ[i_face_2] - ub)/2
            Ψ[i_face_2] = Φ[i_face_2] - (Φ[i_face_1] + Φ[i_face_2] - ub)/2
            # @show cell, w[cell]
            # @show Φ[i_face_1], Φ[i_face_2]
            # @show ub
            # @show Ψ[i_face_1], Ψ[i_face_2]
            if Ψ[i_face_1] < -1e-10
                Ψ[i_face_1] = 0.0
                Ψ[i_face_2] = ub
                # @show Ψ[i_face_1], Ψ[i_face_2]
            elseif Ψ[i_face_2] < -1e-10
                Ψ[i_face_1] = ub
                Ψ[i_face_2] = 0.0
                # @show Ψ[i_face_1], Ψ[i_face_2]
            end
            @assert 1.0 + Ψ[i_face_1] + Ψ[i_face_2] ≈ 1.0 + ub
            @assert -1e-10 <= Ψ[i_face_1] <= a + 1e-10
            @assert -1e-10 <= Ψ[i_face_2] <= b + 1e-10
        elseif Φ[i_face_1] + Φ[i_face_2] < lb
            Ψ[i_face_1] = Φ[i_face_1] - (Φ[i_face_1] + Φ[i_face_2] - lb)/2
            Ψ[i_face_2] = Φ[i_face_2] - (Φ[i_face_1] + Φ[i_face_2] - lb)/2
            # @show cell, w[cell]
            # @show Φ[i_face_1], Φ[i_face_2]
            # @show lb
            # @show Ψ[i_face_1], Ψ[i_face_2]
            if Ψ[i_face_1] > a + 1e-10
                Ψ[i_face_1] = a
                Ψ[i_face_2] = lb - a
                # @show Ψ[i_face_1], Ψ[i_face_2]
            elseif Ψ[i_face_2] > b + 1e-10
                Ψ[i_face_1] = lb - b
                Ψ[i_face_2] = b
                # @show Ψ[i_face_1], Ψ[i_face_2]
            end
            @assert 1.0 + Ψ[i_face_1] + Ψ[i_face_2] ≈ 1.0 + lb
            @assert -1e-10 <= Ψ[i_face_1] <= a + 1e-10
            @assert -1e-10 <= Ψ[i_face_2] <= b + 1e-10
        end
    end
    return Ψ
end

fix(x) = min(1.0, max(0.0, x))

flux = LinearAdvectionFlux([1.0, 1.0])
mesh = PeriodicCartesianMesh(40, 40)
w = map(x -> (x[1]-0.5)^2 + (x[2]-0.5)^2 <= 0.3^2 ? 1.0 : 0.0, cell_centers(mesh))
# w = map(x -> max(abs(x[1]-0.5), abs(x[2]-0.5)) <= 0.25 ? 1.0 : 0.0, cell_centers(mesh))
dt = 0.2 * cell_volume(mesh, CartesianIndex(1, 1)) / face_area(mesh, (Half(1), Half(1)))
scheme = Downwind()

using Plots
cmap_scale_func(n) =  x -> x < 0.5 ? 0.5 - (1 - 2x)^(1/n)/2 : (2x - 1)^(1/n)/2 + 0.5
# anim = @animate for i in 1:400
for i in 1:400
    # @show i
    global w
    @. w = fix(w)

    # Φ = numerical_fluxes(flux, mesh, w, scheme, dt)
    # fcp = positivity_flagged_cells(flux, mesh, w, Φ, dt)
    # fcl = lvd_flagged_cells(flux, mesh, w, Φ, dt)
    # update!(w, Φ, mesh, dt)

    Φ = numerical_fluxes(flux, mesh, w, scheme, dt)
    Ψ = lvd_correction(flux, mesh, w, Φ, dt)
    # fcp = positivity_flagged_cells(flux, mesh, w, Ψ, dt)
    # fcl = lvd_flagged_cells(flux, mesh, w, Ψ, dt)
    update!(w, Ψ, mesh, dt)

    heatmap(w', seriescolor=:viridis,
            # clims=(0, 1),
            aspect_ratio=1,
            title="time step $i, t=$(floor(Int, i*dt*100)/100)",
           ) |> display
    # color=cgrad(:viridis, scale=cmap_scale_func(3)),
    # scatter!(Tuple.(findall(fcp)), marker=:circle, color=:pink, markersize=2, label="positivity") |> display
    # scatter!(Tuple.(findall(fcl)), marker=:cross, color=:red, markersize=2, label="lvd") |> display
end
# gif(anim, "lvd.gif", fps=25)
