#= Optimize a pulse to find the ground-state energy of a molecular Hamiltonian. =#

##########################################################################################
#= PREAMBLE =#
import LinearAlgebra, NPZ
import Plots

mole = "H4"
cE_list = []
REF_list = []
FCI_list = []
for dist in 4:29
    # LOAD MATRIX AND EXTRACT REFERENCE STATES
    matrix = "pennylane_$(mole)_sto-3g_singlet_$(Float64(dist/10))_P-m"      # MATRIX FILE
    H = NPZ.npzread("$(@__DIR__)/matrix/$matrix.npy")

    Λ, U = LinearAlgebra.eigen(LinearAlgebra.Hermitian(H))

    FCI = Λ[1]
    REF = min(Float64.(LinearAlgebra.diag(H))...)

    cE = REF - FCI
    
    push!(cE_list,cE)
    push!(REF_list,REF)
    push!(FCI_list,FCI)
end

plot = Plots.plot()
Plots.plot!(plot, [Float64(dist/10) for dist in 4:29], FCI_list,
    color = :red)
Plots.plot!(plot, [Float64(dist/10) for dist in 4:29], REF_list,
    color = :blue)

Plots.plot!(plot, [Float64(dist/10) for dist in 1:29], cE_list)


