import ..Parameters, ..Devices
export TunableCouplerTransmonDevice, FixedFrequencyTransmonDevice

import ..LinearAlgebraTools
import ..Integrations, ..Signals

import ..Signals: SignalType
import ..LinearAlgebraTools: MatrixList
import ..Quples: Quple

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

using Memoization: @memoize
using LinearAlgebra: I, mul!

#=

The layout of this file includes a `TunableCouplerTransmonDevice` interface,
    and a couple concrete types implementing it.

The purpose of the interface is to minimize code duplication for very similar devices,
    but it is all rather more complicated and ugly than it needs to be.
I suspect a better practice would be to implement each concrete type independently,
    definitely in its own file,
    and probably in its own module.

Therefore, I don't recommend looking too closely to this file as a model to emulate.

=#

abstract type AbstractTunableCouplerNoLocalDriveTransmonDevice{F,FΩ} <: Devices.OnlyNonlocallyDrivenDevice{F,FΩ} end

# THE INTERFACE TO IMPLEMENT

# Devices.nlevels
# Devices.nqubits
# Devices.resonancefrequency
anharmonicity(::AbstractTunableCouplerNoLocalDriveTransmonDevice, q::Int)::Real = error("Not Implemented")

ncouplings(::AbstractTunableCouplerNoLocalDriveTransmonDevice)::Int = error("Not Implemented")
couplingpair(::AbstractTunableCouplerNoLocalDriveTransmonDevice, k::Int)::Quple = error("Not Implemented")
couplingstrength(::AbstractTunableCouplerNoLocalDriveTransmonDevice, k::Int)::Real = error("Not Implemented")

# Devices.ndrives
# Devices.drivequbit
# Devices.drivefrequency
# Devices.drivesignal

bindfrequencies(::AbstractTunableCouplerNoLocalDriveTransmonDevice, ν̄::AbstractVector) = error("Not Implemented")


# THE INTERFACE ALREADY IMPLEMENTED

function Devices.ngrades(device::AbstractTunableCouplerNoLocalDriveTransmonDevice)
    return Devices.ncouplingdrives(device)
end

function Devices.gradequbit(device::AbstractTunableCouplerNoLocalDriveTransmonDevice, j::Int)
    return Devices.drivequbit(device, ((j-1) >> 1) + 1)
end

Devices.eltype_localloweringoperator(::AbstractTunableCouplerNoLocalDriveTransmonDevice{F,FΩ}) where {F,FΩ} = F
function Devices.localloweringoperator(
    device::AbstractTunableCouplerNoLocalDriveTransmonDevice{F,FΩ};
    result=nothing,
) where {F,FΩ}
    isnothing(result) && return _cachedloweringoperator(device)
    result .= 0

    m = Devices.nlevels(device)
    for i ∈ 1:m-1
        result[i,i+1] = √i
    end
    return result
end

@memoize Dict function _cachedloweringoperator(
    device::AbstractTunableCouplerNoLocalDriveTransmonDevice{F,FΩ},
) where {F,FΩ}
    m = Devices.nlevels(device)
    result = Matrix{F}(undef, m, m)
    return Devices.localloweringoperator(device; result=result)
end

Devices.eltype_qubithamiltonian(::AbstractTunableCouplerNoLocalDriveTransmonDevice{F,FΩ}) where {F,FΩ} = F
function Devices.qubithamiltonian(
    device::AbstractTunableCouplerNoLocalDriveTransmonDevice,
    ā::MatrixList,
    q::Int;
    result=nothing,
)
    a = @view(ā[:,:,q])
    Im = Matrix(I, size(a))     # UNAVOIDABLE ALLOCATION?

    result === nothing && (result = Matrix{eltype(a)}(undef, size(a)))
    result .= 0
    result .-= (anharmonicity(device,q)/2)  .* Im           #       - δ/2    I
    result = LinearAlgebraTools.rotate!(a', result)         #       - δ/2   a'a
    result .+= Devices.resonancefrequency(device,q) .* Im    # ω     - δ/2   a'a
    result = LinearAlgebraTools.rotate!(a', result)         # ω a'a - δ/2 a'a'aa
    return result
end

Devices.eltype_coupling(::AbstractTunableCouplerNoLocalDriveTransmonDevice{F,FΩ}) where {F,FΩ} = F
function Devices.couplingoperator(
    device::AbstractTunableCouplerNoLocalDriveTransmonDevice,
    ā::MatrixList{F},
    t::Real;
    result=nothing,
) where {F}
    d = size(ā,1)
    result === nothing && (result = Matrix{F}(undef, d, d))
    aTa = array(F, size(result), LABEL)

    result .= 0
    for pq in 1:ncouplings(device)
        g = Signals.valueat(Devices.couplingdrivesignal(device, pq), t)
        p, q = couplingpair(device, pq)

        aTa = mul!(aTa, (@view(ā[:,:,p]))', @view(ā[:,:,q]))
        result .+= g .* aTa
        result .+= g .* aTa'
    end
    return result
end

function Devices.couplingoperatorwostrengthbyindex(
    device::AbstractTunableCouplerNoLocalDriveTransmonDevice,
    ā::MatrixList{F},
    pq::Int64;
    result=nothing,
) where {F}
    d = size(ā,1)
    result === nothing && (result = Matrix{F}(undef, d, d))
    aTa = array(F, size(result), LABEL)

    result .= 0
    
    p, q = couplingpair(device, pq)

    aTa = mul!(aTa, (@view(ā[:,:,p]))', @view(ā[:,:,q]))
    result .+= aTa
    result .+= aTa'

    return result
end

Devices.eltype_driveoperator(::AbstractTunableCouplerNoLocalDriveTransmonDevice{F,FΩ}) where {F,FΩ} = Complex{F}
function Devices.driveoperator(
    device::AbstractTunableCouplerNoLocalDriveTransmonDevice,
    ā::MatrixList,
    i::Int,
    t::Real;
    result=nothing,
)
    a = @view(ā[:,:,Devices.drivequbit(device, i)])
    e = exp(im * Devices.drivefrequency(device, i) * t)
    Ω = Signals.valueat(Devices.drivesignal(device, i), t)

    if result === nothing
        F = promote_type(eltype(a), eltype(e))  # Ω is no more complex than e.
        result = Matrix{F}(undef, size(a))
    end
    result .= 0

    result .+= (real(Ω) * e ) .* a
    result .+= (real(Ω) * e') .* a'

    if Ω isa Complex
        result .+= (imag(Ω) * im *e ) .* a
        result .+= (imag(Ω) * im'*e') .* a'
    end

    return result
end

Devices.eltype_gradeoperator(::AbstractTunableCouplerNoLocalDriveTransmonDevice{F,FΩ}) where {F,FΩ} = Complex{F}
function Devices.gradeoperator(
    device::AbstractTunableCouplerNoLocalDriveTransmonDevice,
    ā::MatrixList,
    j::Int,
    t::Real;
    result=nothing,
)
    i = ((j-1) >> 1) + 1
    a = @view(ā[:,:,Devices.drivequbit(device, i)])
    e = exp(im * Devices.drivefrequency(device, i) * t)

    if result === nothing
        F = promote_type(eltype(a), eltype(e))
        result = Matrix{F}(undef, size(a))
    end
    result .= 0

    phase = Bool(j & 1) ? 1 : im    # Odd j -> "real" gradient operator; even j  -> "imag"
    result .+= (phase * e ) .* a
    result .+= (phase'* e') .* a'
    return result
end

function Devices.couplinggradeoperator(
    device::AbstractTunableCouplerNoLocalDriveTransmonDevice,
    ā::MatrixList,
    j::Int;
    result=nothing,
)   
    p, q = couplingpair(device, j)

    if result === nothing
        F = promote_type(eltype(ā))
        d = size(ā,1)
        result === nothing && (result = Matrix{F}(undef, d, d))
    end

    aTa = array(F, size(result), LABEL)
    aTa = mul!(aTa, (@view(ā[:,:,p]))', @view(ā[:,:,q]))

    result .= 0
    result .+= aTa
    result .+= aTa'

    return result
end

function Devices.gradient(
    device::AbstractTunableCouplerNoLocalDriveTransmonDevice{F,FΩ},
    grid::Integrations.IntegrationType,
    ϕ̄::AbstractMatrix;
    result=nothing,
) where {F,FΩ}
    L = Parameters.count(device)
    nD = Devices.ndrives(device)
    isnothing(result) && return Devices.gradient(
        device, grid, ϕ̄;
        result=Vector{F}(undef, L),
    )

    gradient_for_signals!(@view(result[1:L-nD]), device, grid, ϕ̄)
    gradient_for_frequencies!(@view(result[1+L-nD:L]), device, grid, ϕ̄)

    return result
end

function gradient_for_signals!(
    result::AbstractVector{F},
    device::AbstractTunableCouplerNoLocalDriveTransmonDevice{F,FΩ},
    grid::Integrations.IntegrationType,
    ϕ̄::AbstractMatrix,
) where {F,FΩ}
    # CALCULATE GRADIENT FOR SIGNAL PARAMETERS
    t̄ = Integrations.lattice(grid)
    ∂̄ = array(FΩ, size(t̄), LABEL)
    Φ = (t, ∂, ϕα, ϕβ) -> (real(∂)*ϕα + imag(∂)*ϕβ)
    Φc = (t, ∂, ϕα) -> (∂*ϕα)

    offset = 0
    for i in 1:Devices.ndrives(device)
        j = 2i - 1
        ϕ̄α = @view(ϕ̄[:,j])
        ϕ̄β = @view(ϕ̄[:,j+1])

        signal = Devices.drivesignal(device, i)
        L = Parameters.count(signal)

        for k in 1:L
            ∂̄ = Signals.partial(k, signal, t̄; result=∂̄)
            result[offset+k] = Integrations.integrate(grid, Φ, ∂̄, ϕ̄α, ϕ̄β)
        end
        offset += L
    end

    for i in 1:Devices.ncouplingdrives(device)
        j = i + 2 * Devices.ndrives(device)
        ϕ̄α = @view(ϕ̄[:,j])

        signal = Devices.couplingdrivesignal(device, i)
        L = Parameters.count(signal)

        for k in 1:L
            ∂̄ = Signals.partial(k, signal, t̄; result=∂̄)
            result[offset+k] = Integrations.integrate(grid, Φc, ∂̄, ϕ̄α)
        end
        offset += L
    end

    return result
end

function gradient_for_frequencies!(
    result::AbstractVector{F},
    device::AbstractTunableCouplerNoLocalDriveTransmonDevice{F,FΩ},
    grid::Integrations.IntegrationType,
    ϕ̄::AbstractMatrix,
) where {F,FΩ}
    # CALCULATE GRADIENT FOR FREQUENCY PARAMETERS
    t̄ = Integrations.lattice(grid)
    Ω̄ = array(FΩ, size(t̄), LABEL)
    Φ = (t, Ω, ϕα, ϕβ) -> (t * (real(Ω)*ϕβ - imag(Ω)*ϕα))

    for i in 1:Devices.ndrives(device)
        j = 2i - 1
        ϕ̄α = @view(ϕ̄[:,j])
        ϕ̄β = @view(ϕ̄[:,j+1])

        signal = Devices.drivesignal(device, i)
        Ω̄ = Signals.valueat(signal, t̄; result=Ω̄)
        result[i] = Integrations.integrate(grid, Φ, Ω̄, ϕ̄α, ϕ̄β)
    end

    return result
end

function Parameters.count(device::AbstractTunableCouplerNoLocalDriveTransmonDevice)
    cnt = Devices.ndrives(device)           # NOTE: There are `ndrives` frequencies.
    for i in 1:Devices.ndrives(device)
        cnt += Parameters.count(Devices.drivesignal(device, i))::Int
    end

    for i in 1:Devices.ncouplingdrives(device)
        cnt += Parameters.count(Devices.couplingdrivesignal(device, i))::Int
    end
    return cnt
end

function Parameters.names(device::AbstractTunableCouplerNoLocalDriveTransmonDevice)
    names = []

    # STRING TOGETHER PARAMETER NAMES FOR EACH SIGNAL Ω̄[i]
    annotate(name,i) = "Ω$i(q$(device.q̄[i])):$name"
    for i in 1:Devices.ndrives(device)
        Ω = Devices.drivesignal(device, i)
        append!(names, (annotate(name,i) for name in Parameters.names(Ω)))
    end

    # TACK ON PARAMETER NAMES FOR EACH ν̄[i]
    append!(names, ("ν$i" for i in 1:Devices.ndrives(device)))
    return names
end

function Parameters.values(device::AbstractTunableCouplerNoLocalDriveTransmonDevice{F,FΩ}) where {F,FΩ}
    values = F[]

    # STRING TOGETHER PARAMETERS FOR EACH SIGNAL Ω̄[i]
    for i in 1:Devices.ndrives(device)
        Ω = Devices.drivesignal(device, i)
        append!(values, Parameters.values(Ω)::Vector{F})
    end

    for pq in 1:ncouplings(device)
        g = Devices.couplingdrivesignal(device,pq)
        append!(values, Parameters.values(g)::Vector{F})
    end
    # TACK ON PARAMETERS FOR EACH ν̄[i]
    append!(values, (Devices.drivefrequency(device,i) for i in 1:Devices.ndrives(device)))
    return values
end

function Parameters.bind(device::AbstractTunableCouplerNoLocalDriveTransmonDevice, x̄::AbstractVector{F}) where {F}
    offset = 0

    # BIND PARAMETERS FOR EACH SIGNAL Ω̄[i]
    for i in 1:Devices.ndrives(device)
        Ω = Devices.drivesignal(device, i)
        L = Parameters.count(Ω)::Int
        Parameters.bind(Ω, x̄[offset+1:offset+L])
        offset += L
    end

    for i in 1:Devices.ncouplingdrives(device)
        g = Devices.couplingdrivesignal(device, i)
        L = Parameters.count(g)::Int
        Parameters.bind(g, x̄[offset+1:offset+L])
        offset += L
    end

    # BIND PARAMETERS FOR EACH ν̄[i]
    bindfrequencies(device, x̄[offset+1:end])
end


"""
    FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice(ω̄, δ̄, quples, q̄, ν̄, Ω̄, m)

A transmon device, modeling for example IBM's superconducting quantum computers.

Variational parameters include the shape parameters in each pulse and the 
    interqubit coupling strengths.
    Pulse frequencies are "frozen".

# Arguments
- `ω̄`: a vector of angular frequencies specifying the resonance frequency of each qubit.
- `δ̄`: a vector of angular frequencies specifying the anharmonicity of each qubit.

- `ḡ`: a vector of angular frequencies specifying the strength of each coupling.
- `quples`: a vector of `Quple` identifying whcih qubits participate in each coupling.

- `q̄`: a vector of indices specifying the target qubit for each drive channel.
- `ν̄`: a vector of angular frequencies specifying the pulse frequencies for each channel.
- `Ω̄`: a vector of signals specifying the shape of the pulse for each channel.

- `m`: an integer specifying the number of physical levels to retain for each qubit.

"""
struct FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice{F,FΩ} <: AbstractTunableCouplerNoLocalDriveTransmonDevice{F,FΩ}
    # QUBIT LISTS
    ω̄::Vector{F}
    δ̄::Vector{F}
    # COUPLING LISTS
    quples::Vector{Quple}
    # DRIVE LISTS
    q̄::Vector{Int}
    ν̄::Vector{F}
    Ω̄::Vector{SignalType{F,FΩ}}
    ḡ::AbstractVector{<:SignalType{F,F}}
    # OTHER PARAMETERS
    m::Int

    function FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice(
        ω̄::AbstractVector{<:Real},
        δ̄::AbstractVector{<:Real},
        quples::AbstractVector{Quple},
        q̄::AbstractVector{Int},
        ν̄::AbstractVector{<:AbstractFloat},
        Ω̄::AbstractVector{<:SignalType{F,FΩ}},
        ḡ::AbstractVector{<:SignalType{F,F}},
        m::Int,
    ) where {F,FΩ}
        # VALIDATE PARALLEL LISTS ARE CONSISTENT SIZE
        @assert length(ω̄) == length(δ̄) ≥ 1              # NUMBER OF QUBITS
        @assert length(q̄) == length(ν̄) == length(Ω̄)     # NUMBER OF DRIVES

        # VALIDATE QUBIT INDICES
        for (p,q) in quples
            @assert 1 <= p <= length(ω̄)
            @assert 1 <= q <= length(ω̄)
        end
        for q in q̄
            @assert 1 <= q <= length(ω̄)
        end

        # VALIDATE THAT THE HILBERT SPACE HAS SOME VOLUME...
        @assert m ≥ 2

        # STANDARDIZE TYPING
        return new{F,FΩ}(
            convert(Vector{F}, ω̄),
            convert(Vector{F}, δ̄),
            quples,
            q̄,
            convert(Vector{F}, ν̄),
            [Ω for Ω in Ω̄],
            [g for g in ḡ],
            m,
        )
    end
end

Devices.nlevels(device::FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice) = device.m

Devices.nqubits(device::FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice) = length(device.ω̄)
Devices.resonancefrequency(device::FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice, q::Int) = device.ω̄[q]
anharmonicity(device::FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice, q::Int) = device.δ̄[q]

ncouplings(device::FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice) = length(device.quples)
couplingpair(device::FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice, k::Int) = device.quples[k]

Devices.ndrives(device::FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice) = 0
Devices.ncouplingdrives(device::FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice) = ncouplings(device)
Devices.drivequbit(device::FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice, i::Int) = device.q̄[i]
Devices.drivefrequency(device::FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice, i::Int) = device.ν̄[i]
Devices.__get__drivesignals(device::FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice) = []
Devices.__get__couplingdrivesignals(device::FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice) = device.ḡ

bindfrequencies(device::FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice, ν̄::AbstractVector) = (device.ν̄ .= ν̄)

function Parameters.count(device::FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice)
    cnt = 0       

    for i in 1:Devices.ncouplingdrives(device)
        cnt += Parameters.count(Devices.couplingdrivesignal(device, i))::Int
    end
    return cnt
end

function Parameters.names(device::FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice)
    names = []

    # STRING TOGETHER PARAMETER NAMES FOR EACH SIGNAL Ω̄[i]
    annotate(name,i) = "Ω$i(q$(device.q̄[i])):$name"
    for i in 1:Devices.ndrives(device)
        Ω = Devices.drivesignal(device, i)
        append!(names, (annotate(name,i) for name in Parameters.names(Ω)))
    end

    return names
end

function Parameters.values(device::FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice{F,FΩ}) where {F,FΩ}
    values = F[]

    for pq in 1:ncouplings(device)
        g = Devices.couplingdrivesignal(device,pq)
        append!(values, Parameters.values(g)::Vector{F})
    end
    return values
end

function Parameters.bind(device::FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice, x̄::AbstractVector{F}) where {F}
    offset = 0

    for i in 1:Devices.ncouplingdrives(device)
        g = Devices.couplingdrivesignal(device, i)
        L = Parameters.count(g)::Int
        Parameters.bind(g, x̄[offset+1:offset+L])
        offset += L
    end
end

function Devices.gradient(
    device::FixedFrequencyTunableCouplerNoLocalDriveTransmonDevice{F,FΩ},
    grid::Integrations.IntegrationType,
    ϕ̄::AbstractMatrix;
    result=nothing,
) where {F,FΩ}
    L = Parameters.count(device)
    nD = Devices.ndrives(device)
    isnothing(result) && return Devices.gradient(
        device, grid, ϕ̄;
        result=Vector{F}(undef, L),
    )

    gradient_for_signals!(@view(result[1:L]), device, grid, ϕ̄)

    return result
end