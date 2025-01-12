#= Optimize a pulse to find the ground-state energy of a molecular Hamiltonian. =#

##########################################################################################
#= PREAMBLE =#
import CtrlVQE
import Random, LinearAlgebra
import NPZ, Optim, LineSearches
import Unicode: ispunct
import JLD2: load,save

global seedStr = ARGS[1]
global distStr = ARGS[2]
global TStr = ARGS[3]
global moleStr = ARGS[4]
global WStr = ARGS[5]
global rStr = ARGS[6]
global initgStr = ARGS[7]
global maxiterStr = ARGS[8]
global dirStr = ARGS[9]

# global seedStr = "1"
# global distStr = "15"
# global TStr = "30"
# global moleStr = "H4"
# global WStr = "50"
# global rStr = "1000"
# global initgStr = "0.002"
# global maxiterStr = "10000"
# global dirStr = "/Users/gxc/Documents/Projects/9_CtrlVQE_TunableCoupler/data"

println("args: 1: $seedStr, 2: $distStr, 3: $TStr, 4: $moleStr, 5: $WStr, 6: $rStr, 7: $initgStr, 8: $maxiterStr, 9: $dirStr")

dist = parse(Float64, distStr)/10
mole = moleStr

# PARENT DIRECTORY
dir =  dirStr         
T = parse(Float64, TStr) # ns           # TOTAL DURATION OF PULSE
W = parse(Int64, WStr)                  # NUMBER OF WINDOWS IN EACH PULSE

r = parse(Int64,rStr)                   # NUMBER OF STEPS IN TIME EVOLUTION
m = 2                                   # NUMBER OF LEVELS PER TRANSMON

seed = parse(Int64,seedStr)             # RANDOM SEED FOR PULSE INTIALIZATION
init_Ω = 0.0 # 2π GHz                   # AMPLITUDE RANGE FOR PULSE INITIALIZATION
init_φ = 0.0                            # PHASE RANGE FOR PULSE INITIALIZATION
init_Δ = 0.0 # 2π GHz                   # FREQUENCY RANGE FOR PULSE INITIALIZATION

ΩMAX = 2π * 0.02 # 2π GHz               # LOCAL DRIVE AMPLITUDE BOUNDS
λΩ = 1.0 # Ha                           # PENALTY WEIGHT FOR EXCEEDING AMPLITUDE BOUNDS
σΩ = ΩMAX                               # PENALTY STEEPNESS FOR EXCEEDING AMPLITUDE BOUNDS

ΔMAX = 2π * 1.00 # 2π GHz               # FREQUENCY BOUNDS
λΔ = 1.0 # Ha                           # PENALTY WEIGHT FOR EXCEEDING FREQUENCY BOUNDS
σΔ = ΔMAX                               # PENALTY STEEPNESS FOR EXCEEDING FREQUENCY BOUNDS

f_tol = 0.0                             # TOLERANCE IN FUNCTION EVALUATION
g_tol = 1e-6                            # TOLERANCE IN GRADIENT NORM
maxiter = parse(Int64,maxiterStr)       # MAXIMUM NUMBER OF ITERATIONS

##########################################################################################
#= SETUP =#

# LOAD MATRIX AND EXTRACT REFERENCE STATES
matrix = "pennylane_$(mole)_sto-3g_singlet_$(dist)_P-m"      # MATRIX FILE
H = NPZ.npzread("$(@__DIR__)/matrix/$matrix.npy")

# matrix2 = "H2_sto-3g_singlet_$(dist)_P-m"     # ALTERNATIVE MATRIX FILE
# H2 = NPZ.npzread("$(@__DIR__)/matrix/$matrix2.npy")

# matrix3 = "h2$(filter(!ispunct, string(dist)))"                     # ALTERNATIVE MATRIX FILE
# H3 = NPZ.npzread("$(@__DIR__)/matrix/$matrix3.npy")
n = CtrlVQE.QubitOperators.nqubits(H)
ψ_REF = CtrlVQE.QubitOperators.reference(H)         # REFERENCE STATE
REF = real(ψ_REF' * H * ψ_REF)                      # REFERENCE STATE ENERGY

# IDENTIFY EXACT RESULTS
Λ, U = LinearAlgebra.eigen(LinearAlgebra.Hermitian(H))
ψ_FCI = U[:,1]                              # GROUND STATE
FCI = Λ[1]                                  # GROUND STATE ENERGY
FES = Λ[2]                                  # FIRST EXCITED STATE

# CONSTRUCT THE MAJOR PACKAGE OBJECTS

grid = CtrlVQE.TemporalLattice(T, r)

pulse = CtrlVQE.UniformWindowed(CtrlVQE.ComplexConstant(0.0, 0.0), T, W); ΩMAX /= √2
            # NOTE: Re-scale max amplitude so that bounds inscribe the complex circle.
            #       Not needed for real or polar-parameterized amplitudes.
# gpulse = CtrlVQE.UniformWindowed(CtrlVQE.Constant(0.02), T, W)
# pulse = CtrlVQE.UniformWindowed(CtrlVQE.PolarComplexConstant(0.0, 0.0), T, W)

# device = CtrlVQE.SystematicTunable(CtrlVQE.TunableCouplerTransmonDevice, n, pulse, gpulse)
# device = CtrlVQE.Systematic(CtrlVQE.TransmonDevice, n, pulse)
device = CtrlVQE.Systematic(CtrlVQE.FixedFrequencyTransmonDevice, n, pulse)

evolution = CtrlVQE.TOGGLE

# INITIALIZE PARAMETERS
Random.seed!(seed)
xi = CtrlVQE.Parameters.values(device)

L = length(xi)                              # NUMBER OF PARAMETERS
Ω = 1:L; φ = []; ν = []                     # INDEXING VECTORS (Cartesian sans Frequencies)
# Ω = 1:L-n; φ = []; ν = 1+L-n:L            # INDEXING VECTORS (Cartesian with Frequencies)
# Ω = 1:2:L-n; φ = 2:2:L-n; ν = 1+L-n:L     # INDEXING VECTORS (Polar with Frequenices)
# Ω = 1:2:L; φ = 2:2:L; ν = []              # INDEXING VECTORS (Polar sans Frequencies)

xi[Ω] .+= init_Ω .* (2 .* rand(length(Ω)) .- 1)
xi[φ] .+= init_φ .* (2 .* rand(length(φ)) .- 1)
xi[ν] .+= init_Δ .* (2 .* rand(length(ν)) .- 1)

##########################################################################################
#= PREPARE OPTIMIZATION OBJECTS =#

# ENERGY FUNCTIONS
O0 = CtrlVQE.QubitOperators.project(H, device)              # MOLECULAR HAMILTONIAN
ψ0 = CtrlVQE.QubitOperators.project(ψ_REF, device)          # REFERENCE STATE

fn_energy = CtrlVQE.ProjectedEnergy(
    evolution, device,
    CtrlVQE.OCCUPATION, CtrlVQE.STATIC,
    grid, ψ0, O0,
)

# PENALTY FUNCTIONS
λ  = zeros(L);  λ[Ω] .=    λΩ                               # PENALTY WEIGHTS
μR = zeros(L); μR[Ω] .= +ΩMAX                               # PENALTY LOWER BOUNDS
μL = zeros(L); μL[Ω] .= -ΩMAX                               # PENALTY UPPER BOUNDS
σ  = zeros(L);  σ[Ω] .=    σΩ                               # PENALTY SCALINGS
fn_penalty = CtrlVQE.SmoothBound(λ, μR, μL, σ)

# CALLBACK FUNCTION
trace = Dict("energy" => [], "g_norm" => [])
cb = tr -> begin
            push!(trace["g_norm"], tr[end].g_norm)
            push!(trace["energy"], tr[end].value)
            false
        end

# OPTIMIZATION FUNCTIONS
fn_total = CtrlVQE.CompositeCostFunction(fn_energy, fn_penalty)
f  = CtrlVQE.cost_function(fn_total)
g! = CtrlVQE.grad_function_inplace(fn_total)


# OPTIMIZATION ALGORITHM
linesearch = LineSearches.MoreThuente()
optimizer = Optim.LBFGS(linesearch=linesearch)

# OPTIMIZATION OPTIONS
options = Optim.Options(
    show_trace = true,
    show_every = 1,
    f_tol = f_tol,
    g_tol = g_tol,
    iterations = maxiter,
    extended_trace=false,
    store_trace=true,
    callback = cb
)

##########################################################################################
#= DATA DIRECTORY AND FILE NAME=#
fn = "trace.$mole.fixedcoupling.W.$W.T.$T.r.$r.maxiter.$maxiter.m.$m.dist.$dist.jld2"
subdir = "$dir/molecular_hamiltonian/fixedcoupling/$(mole)_$(dist)"
fn_fp = "$subdir/$fn"
if isdir(subdir)
    if isfile(fn_fp)
        trace_load = load(fn_fp)
        print("Done before: $fn")
        exit()
    end
else
    mkpath(subdir)
end
##########################################################################################
#= RUN OPTIMIZATION =#

optimization = Optim.optimize(f, g!, xi, optimizer, options)
xf = Optim.minimizer(optimization)      # FINAL PARAMETERS

##########################################################################################
#= REPORT RESULTS =#

ff = fn_total(xf)       # LOSS FUNCTION
Ef = fn_energy(xf)      # CURRENT ENERGY
λf = fn_penalty(xf)     # PENALTY CONTRIBUTION
εE = Ef - FCI           # ENERGY ERROR
cE = 1-εE/(REF-FCI)     # CORRELATION ENERGY

Ωsat = sum((
    count(xf[Ω] .≥ ( ΩMAX)),
    count(xf[Ω] .≤ (-ΩMAX)),
))

println("""

    Optimization Results
    --------------------
     Energy Error: $εE
    % Corr Energy: $(cE*100)

    Loss Function: $ff
        from  Energy: $Ef
        from Penalty: $λf

    Saturated Amplitudes: $Ωsat / $(length(Ω))
""")


##########################################################################################
#= PLOT RESULTS =#

# EXTRACT REAL/IMAGINARY PARTS OF THE PULSE
# t = CtrlVQE.lattice(grid)
# CtrlVQE.Parameters.bind(device, xf)             # ENSURE DEVICE USES THE FINAL PARAMETERS
# nD = CtrlVQE.ndrives(device)
# α = Array{Float64}(undef, r+1, nD)
# β = Array{Float64}(undef, r+1, nD)
# for i in 1:nD
#     Ωt = CtrlVQE.Devices.drivesignal(device, i)(t)
#     α[:,i] = real.(Ωt)
#     β[:,i] = imag.(Ωt)
# end

# # SET UP PLOT OBJECT
# yMAX = (ΩMAX / 2π) * 1.1        # Divide by 2π to convert angular frequency to frequency.
#                                 # Multiply by 1.1 to add a little buffer to the plot.
# plot = Plots.plot(;
#     xlabel= "Time (ns)",
#     ylabel= "|Amplitude| (GHz)",
#     ylims = [-yMAX, yMAX],
#     legend= :topright,
# )

# # DUMMY PLOT OBJECTS TO SETUP LEGEND THE WAY WE WANT IT
# Plots.plot!(plot, [0], [2yMAX], lw=3, ls=:solid, color=:black, label="α")
# Plots.plot!(plot, [0], [2yMAX], lw=3, ls=:dot, color=:black, label="β")

# # PLOT AMPLITUDES
# for i in 1:nD
#     Plots.plot!(plot, t, α[:,i]./2π, lw=3, ls=:solid, color=i, label="Drive $i")
#     Plots.plot!(plot, t, β[:,i]./2π, lw=3, ls=:dot, color=i, label=false)
# end
# display(plot)

trace = merge(trace, Dict("xf" => xf, "Ef" => Ef, "energy error" => εE, "corr energy" => (cE*100)))

if isdir(subdir)
    save(fn_fp,trace)
end