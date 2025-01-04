#= Optimize a pulse to find the ground-state energy of a molecular Hamiltonian. =#

##########################################################################################
#= PREAMBLE =#
import CtrlVQE
import Random, LinearAlgebra
import NPZ, Optim, LineSearches, Plots

matrix = "H2_sto-3g_singlet_1.5_P-m"    # MATRIX FILE
T = 5.0 # ns                # TOTAL DURATION OF PULSE
W = 10                      # NUMBER OF WINDOWS IN EACH PULSE

r = round(Int,20T)          # NUMBER OF STEPS IN TIME EVOLUTION
m = 2                       # NUMBER OF LEVELS PER TRANSMON

seed = 9999                 # RANDOM SEED FOR PULSE INTIALIZATION
init_Ω = 0.0 # 2π GHz       # AMPLITUDE RANGE FOR PULSE INITIALIZATION
init_φ = 0.0                # PHASE RANGE FOR PULSE INITIALIZATION
init_Δ = 0.0 # 2π GHz       # FREQUENCY RANGE FOR PULSE INITIALIZATION
init_g = 0.02                # AMPLITUDE RANGE FOR COUPLING PULSE INITIALIZATION

ΩMAX = 2π * 0.02 # 2π GHz   # LOCAL DRIVE AMPLITUDE BOUNDS
λΩ = 1.0 # Ha               # PENALTY WEIGHT FOR EXCEEDING AMPLITUDE BOUNDS
σΩ = ΩMAX                   # PENALTY STEEPNESS FOR EXCEEDING AMPLITUDE BOUNDS
gMAX = 2π * 0.02            # NONLOCAL DRIVE AMPLITUDE BOUNDS

ΔMAX = 2π * 1.00 # 2π GHz   # FREQUENCY BOUNDS
λΔ = 1.0 # Ha               # PENALTY WEIGHT FOR EXCEEDING FREQUENCY BOUNDS
σΔ = ΔMAX                   # PENALTY STEEPNESS FOR EXCEEDING FREQUENCY BOUNDS

f_tol = 0.0                 # TOLERANCE IN FUNCTION EVALUATION
g_tol = 1e-6                # TOLERANCE IN GRADIENT NORM
maxiter = 10000             # MAXIMUM NUMBER OF ITERATIONS

##########################################################################################
#= SETUP =#

# LOAD MATRIX AND EXTRACT REFERENCE STATES
H = NPZ.npzread("$(@__DIR__)/matrix/$matrix.npy")
n = CtrlVQE.QubitOperators.nqubits(H)
ψ_REF = CtrlVQE.QubitOperators.reference(H) # REFERENCE STATE
REF = real(ψ_REF' * H * ψ_REF)              # REFERENCE STATE ENERGY

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
gpulse = CtrlVQE.UniformWindowed(CtrlVQE.Constant(0.0), T, W)
# pulse = CtrlVQE.UniformWindowed(CtrlVQE.PolarComplexConstant(0.0, 0.0), T, W)
testpulse = CtrlVQE.UniformWindowed(CtrlVQE.ComplexConstant(0.0, 0.0), T, W); ΩMAX /= √2
CtrlVQE.Parameters.bind(testpulse,[rand() for _ in 1:2*W])

testgpulse = CtrlVQE.UniformWindowed(CtrlVQE.Constant(0.0), T, W)
CtrlVQE.Parameters.bind(testgpulse,[rand() for _ in 1:W])

device = CtrlVQE.Systematic(CtrlVQE.FixedFrequencyTransmonDevice, n, pulse)
device2 = CtrlVQE.SystematicTunable(CtrlVQE.TunableCouplerTransmonDevice, n, pulse, gpulse)
# device = CtrlVQE.Systematic(CtrlVQE.TransmonDevice, n, pulse)

evolution = CtrlVQE.TOGGLE
evolution2 = CtrlVQE.TUNABLECOUPLETOGGLE

# INITIALIZE PARAMETERS
Random.seed!(seed)
xi = CtrlVQE.Parameters.values(device)

L = length(xi)                      # NUMBER OF PARAMETERS
Ω = 1:L; φ = []; ν = []                 # INDEXING VECTORS (Cartesian sans Frequencies)
# Ω = 1:L-n; φ = []; ν = 1+L-n:L          # INDEXING VECTORS (Cartesian with Frequencies)
# Ω = 1:2:L-n; φ = 2:2:L-n; ν = 1+L-n:L   # INDEXING VECTORS (Polar with Frequenices)
# Ω = 1:2:L; φ = 2:2:L; ν = []            # INDEXING VECTORS (Polar sans Frequencies)

xi[Ω] .+= init_Ω .* (2 .* rand(length(Ω)) .- 1)
xi[φ] .+= init_φ .* (2 .* rand(length(φ)) .- 1)
xi[ν] .+= init_Δ .* (2 .* rand(length(ν)) .- 1)

xi2 = CtrlVQE.Parameters.values(device2)

L2 = length(xi2)
Ω2 = 1:2*CtrlVQE.Parameters.count(pulse)
g2 = 2*CtrlVQE.Parameters.count(pulse)+1:2*CtrlVQE.Parameters.count(pulse)+CtrlVQE.Parameters.count(gpulse)
φ2 = []; ν2 = 2*CtrlVQE.Parameters.count(pulse)+CtrlVQE.Parameters.count(gpulse)+1:CtrlVQE.Parameters.count(device2)

xi2[Ω2] .+= init_Ω .* (2 .* rand(length(Ω2)) .- 1)
xi2[φ2] .+= init_φ .* (2 .* rand(length(φ2)) .- 1)
xi2[g2] .+= init_g .* (2 .* rand(length(g2)) .- 1)
xi2[ν2] .+= init_Δ .* (2 .* rand(length(ν2)) .- 1)

##########################################################################################
#= PREPARE OPTIMIZATION OBJECTS =#

# STATIC HAMILTONIAN H0 = ∑ⱼ ωⱼ(aⱼ†)aⱼ - δₗ/2 (aⱼ†)(aⱼ†)aⱼaⱼ + ∑ₐᵦ [gₐᵦ(aₐ†)aᵦ+(aᵦ†)aₐ]
# H0 = CtrlVQE.Devices.operator(CtrlVQE.STATIC,device,CtrlVQE.OCCUPATION);
# expiH0T = Matrix{ComplexF64}(undef,size(H));

# function ∂gpqexpjH0t(
#     op::CtrlVQE.Operators.StaticOperator,
#     device::CtrlVQE.DeviceType,
#     basis::CtrlVQE.Bases.BasisType,
#     λ::Float64,
#     i::Int64,
#     t::Real;
#     result=nothing
# )
#     H0 = CtrlVQE.LinearAlgebraTools.cis_type(CtrlVQE.Devices.operator(op, device, basis, :cache))
#     isnothing(result) && (result=Matrix{CtrlVQE.LinearAlgebraTools.cis_type(H)})
#     ā = CtrlVQE.Devices.algebra(device, basis)

#     couplingop = CtrlVQE.Devices.couplingoperatorwostrengthbyindex(device2,ā,i)
#     result = im * λ * t * CtrlVQE.LinearAlgebraTools.cis!(H0,λ * t) * couplingop * CtrlVQE.LinearAlgebraTools.cis!(H0,(1-λ) * t) 
#     return result
# end

# ENERGY FUNCTIONS
O0 = CtrlVQE.QubitOperators.project(H, device)              # MOLECULAR HAMILTONIAN
ψ0 = CtrlVQE.QubitOperators.project(ψ_REF, device)          # REFERENCE STATE

fn_energy = CtrlVQE.ProjectedEnergy(
    evolution, device,
    CtrlVQE.OCCUPATION, CtrlVQE.STATIC,
    grid, ψ0, O0,
)

fn_energy2 = CtrlVQE.ProjectedEnergyTunableCoupler(
    evolution2, device2,
    CtrlVQE.OCCUPATION, CtrlVQE.STATIC,
    grid, ψ0, O0,
)

# PENALTY FUNCTIONS
λ  = zeros(L);  λ[Ω] .=    λΩ                               # PENALTY WEIGHTS
μR = zeros(L); μR[Ω] .= +ΩMAX                               # PENALTY LOWER BOUNDS
μL = zeros(L); μL[Ω] .= -ΩMAX                               # PENALTY UPPER BOUNDS
σ  = zeros(L);  σ[Ω] .=    σΩ                               # PENALTY SCALINGS
fn_penalty = CtrlVQE.SmoothBound(λ, μR, μL, σ)

# PENALTY FUNCTIONS
λ  = zeros(L2);  λ[1:L2] .= λΩ                              # PENALTY WEIGHTS
μR = zeros(L2);                                             # PENALTY UPPER BOUNDS
μR[Ω2] .= +ΩMAX                                             # LOCAL DRIVES UPPER BOUNDS
μR[g2] .= +gMAX                                             # NONLOCAL DRIVES UPPER BOUNDS
μR[ν2] .= device2.ν̄ .+ ΔMAX       # NONLOCAL DRIVES UPPER BOUNDS
μL = zeros(L2);                                             # PENALTY LOWER BOUNDS
μL[Ω2] .= -ΩMAX                                             # LOCAL DRIVES LOWER BOUNDS
μL[ν2] .= device2.ν̄ .- ΔMAX       # LOCAL DRIVES LOWER BOUNDS
μL[g2] .= -gMAX                                             # NONLOCAL DRIVES LOWER BOUNDS

σ  = zeros(L2);  σ[1:L2] .=    σΩ                           # PENALTY SCALINGS
fn_penalty2 = CtrlVQE.SmoothBound(λ, μR, μL, σ)

# OPTIMIZATION FUNCTIONS
fn_total = CtrlVQE.CompositeCostFunction(fn_energy, fn_penalty)
f  = CtrlVQE.cost_function(fn_total)
g! = CtrlVQE.grad_function_inplace(fn_total)

# OPTIMIZATION FUNCTIONS
fn_total2 = CtrlVQE.CompositeCostFunction(fn_energy2, fn_penalty2)
f2  = CtrlVQE.cost_function(fn_total2)
g2! = CtrlVQE.grad_function_inplace(fn_total2)

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
)
##########################################################################################
#= RUN OPTIMIZATION =#

optimization = Optim.optimize(f2, g2!, xi2, optimizer, options)
xf = Optim.minimizer(optimization)      # FINAL PARAMETERS

##########################################################################################
#= REPORT RESULTS =#

ff = fn_total2(xf)       # LOSS FUNCTION
Ef = fn_energy2(xf)      # CURRENT ENERGY
λf = fn_penalty2(xf)     # PENALTY CONTRIBUTION
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
t = CtrlVQE.lattice(grid)
CtrlVQE.Parameters.bind(device, xf)             # ENSURE DEVICE USES THE FINAL PARAMETERS
nD = CtrlVQE.ndrives(device)
α = Array{Float64}(undef, r+1, nD)
β = Array{Float64}(undef, r+1, nD)
for i in 1:nD
    Ωt = CtrlVQE.TransmonDevices.drivesignal(device, i)(t)
    α[:,i] = real.(Ωt)
    β[:,i] = imag.(Ωt)
end

# SET UP PLOT OBJECT
yMAX = (ΩMAX / 2π) * 1.1        # Divide by 2π to convert angular frequency to frequency.
                                # Multiply by 1.1 to add a little buffer to the plot.
plot = Plots.plot(;
    xlabel= "Time (ns)",
    ylabel= "|Amplitude| (GHz)",
    ylims = [-yMAX, yMAX],
    legend= :topright,
)

# DUMMY PLOT OBJECTS TO SETUP LEGEND THE WAY WE WANT IT
Plots.plot!(plot, [0], [2yMAX], lw=3, ls=:solid, color=:black, label="α")
Plots.plot!(plot, [0], [2yMAX], lw=3, ls=:dot, color=:black, label="β")

# PLOT AMPLITUDES
for i in 1:nD
    Plots.plot!(plot, t, α[:,i]./2π, lw=3, ls=:solid, color=i, label="Drive $i")
    Plots.plot!(plot, t, β[:,i]./2π, lw=3, ls=:dot, color=i, label=false)
end

# SHOW PLOT
Plots.gui()