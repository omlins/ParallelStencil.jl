const DATA_DOC = """
    Module Data

The module Data is created in the module where `@init_parallel_kernel` is called from. It provides the following types:

--------------------------------------------------------------------------------
    Data.Number

The type of numbers used by @zeros, @ones, @rand and @fill and in all array types of module `Data` (selected with argument `numbertype` of [`@init_parallel_kernel`](@ref)).

--------------------------------------------------------------------------------
    Data.Array{ndims}

Expands to `Data.Array{numbertype, ndims}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the datatype `Data.Array` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Array for Threads, CUDA.CuArray or CUDA.CuDeviceArray for CUDA and AMDGPU.ROCArray or AMDGPU.ROCDeviceArray for AMDGPU; [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray and AMDGPU.ROCArray automatically to CUDA.CuDeviceArray and AMDGPU.ROCDeviceArray in kernels when required).

--------------------------------------------------------------------------------
    Data.CellArray{ndims}

Expands to `Data.CellArray{numbertype, ndims}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the datatype `Data.CellArray` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (CPUCellArray for Threads, CuCellArray or CuDeviceCellArray for CUDA and ROCCellArray or ROCDeviceCellArray for AMDGPU; [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CellArray automatically to DeviceCellArray when required).

--------------------------------------------------------------------------------
    Data.Cell{S}

Expands to `Union{StaticArrays.SArray{S, numbertype}, StaticArrays.FieldArray{S, numbertype}}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref).

--------------------------------------------------------------------------------
    Data.NumberTuple{N_tuple} | Data.NamedNumberTuple{N_tuple, names} | Data.NumberCollection{N_tuple}

Expands to: `NTuple{N_tuple, Data.Number}` | `NamedTuple{names, NTuple{N_tuple, Data.Number}}` | `Union{Data.NumberTuple{N_tuple}, Data.NamedNumberTuple{N_tuple}}`

--------------------------------------------------------------------------------
    Data.ArrayTuple{N_tuple, N} | Data.NamedArrayTuple{N_tuple, N, names} | Data.ArrayCollection{N_tuple, N}

Expands to: `NTuple{N_tuple, Data.Array{N}}` | `NamedTuple{names, NTuple{N_tuple, Data.Array{N}}}` | `Union{Data.ArrayTuple{N_tuple, N}, Data.NamedArrayTuple{N_tuple, N}}`

--------------------------------------------------------------------------------
    Data.CellArrayTuple{N_tuple, N, B} | Data.NamedCellArrayTuple{N_tuple, N, B, names} | Data.CellArrayCollection{N_tuple, N, B}

Expands to: `NTuple{N_tuple, Data.CellArray{N, B}}` | `NamedTuple{names, NTuple{N_tuple, Data.CellArray{N, B}}}` | `Union{Data.CellArrayTuple{N_tuple, N, B}, Data.NamedCellArrayTuple{N_tuple, N, B}}`

--------------------------------------------------------------------------------
    Data.CellTuple{N_tuple, S} | Data.NamedCellTuple{N_tuple, S, names} | Data.CellCollection{N_tuple, S}

Expands to: `NTuple{N_tuple, Data.Cell{S}}` | `NamedTuple{names, NTuple{N_tuple, Data.Cell{S}}}` | `Union{Data.CellTuple{N_tuple, S}, Data.NamedCellTuple{N_tuple, S}}`

--------------------------------------------------------------------------------
!!! note "Advanced"
        Data.DeviceArray{ndims}

    Expands to `Data.DeviceArray{numbertype, ndims}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the datatype `Data.DeviceArray` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Array for Threads, CUDA.CuDeviceArray for CUDA AMDGPU.ROCDeviceArray for AMDGPU).

    !!! warning
        This datatype is not intended for explicit manual usage. [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray and AMDGPU.ROCArray automatically to CUDA.CuDeviceArray and AMDGPU.ROCDeviceArray in kernels when required.

    --------------------------------------------------------------------------------
        Data.DeviceCellArray{ndims}

    Expands to `Data.DeviceCellArray{numbertype, ndims}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref) and the datatype `Data.DeviceCellArray` is chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (CPUCellArray for Threads, CuDeviceCellArray for CUDA and ROCDeviceCellArray for AMDGPU).

    !!! warning
        This datatype is not intended for explicit manual usage. [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray and AMDGPU.ROCArray automatically to CUDA.CuDeviceArray and AMDGPU.ROCDeviceArray in kernels when required.
"""

const DATA_DOC_NUMBERTYPE_NONE = """
    Module Data

The module Data is created in the module where `@init_parallel_kernel` is called from. It provides the following types:

--------------------------------------------------------------------------------
    Data.Array{numbertype, ndims}

The datatype `Data.Array` is automatically chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Array for Threads, CUDA.CuArray or CUDA.CuDeviceArray for CUDA and AMDGPU.ROCArray or AMDGPU.ROCDeviceArray for AMDGPU; [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray and AMDGPU.ROCArray automatically to CUDA.CuDeviceArray and AMDGPU.ROCDeviceArray in kernels when required).

--------------------------------------------------------------------------------
    Data.CellArray{numbertype, ndims}

The datatype `Data.CellArray` is automatically chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (CPUCellArray for Threads, CuCellArray or CuDeviceCellArray for CUDA and ROCCellArray or ROCDeviceCellArray for AMDGPU; [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CellArray automatically to DeviceCellArray in kernels when required).

--------------------------------------------------------------------------------
    Data.Cell{numbertype, S}

Expands to `Union{StaticArrays.SArray{S, numbertype}, StaticArrays.FieldArray{S, numbertype}}`, where `numbertype` is the datatype selected with [`@init_parallel_kernel`](@ref).

--------------------------------------------------------------------------------
    Data.NumberTuple{N_tuple, numbertype} | Data.NamedNumberTuple{N_tuple, numbertype, names} | Data.NumberCollection{N_tuple, numbertype}

Expands to: `NTuple{N_tuple, numbertype}` | `NamedTuple{names, NTuple{N_tuple, numbertype}}` | `Union{Data.NumberTuple{N_tuple, numbertype}, Data.NamedNumberTuple{N_tuple, numbertype}}`

--------------------------------------------------------------------------------
    Data.ArrayTuple{N_tuple, numbertype, N} | Data.NamedArrayTuple{N_tuple, numbertype, N, names} | Data.ArrayCollection{N_tuple, numbertype, N}

Expands to: `NTuple{N_tuple, Data.Array{numbertype, N}}` | `NamedTuple{names, NTuple{N_tuple, Data.Array{numbertype, N}}}` | `Union{Data.ArrayTuple{N_tuple, numbertype, N}, Data.NamedArrayTuple{N_tuple, numbertype, N}}`

--------------------------------------------------------------------------------
    Data.CellArrayTuple{N_tuple, numbertype, N, B} | Data.NamedCellArrayTuple{N_tuple, numbertype, N, B, names} | Data.CellArrayCollection{N_tuple, numbertype, N, B}

Expands to: `NTuple{N_tuple, Data.CellArray{numbertype, N, B}}` | `NamedTuple{names, NTuple{N_tuple, Data.CellArray{numbertype, N, B}}}` | `Union{Data.CellArrayTuple{N_tuple, numbertype, N, B}, Data.NamedCellArrayTuple{N_tuple, numbertype, N, B}}`

--------------------------------------------------------------------------------
    Data.CellTuple{N_tuple, numbertype, S} | Data.NamedCellTuple{N_tuple, numbertype, S, names} | Data.CellCollection{N_tuple, numbertype, S}

Expands to: `NTuple{N_tuple, Data.Cell{numbertype, S}}` | `NamedTuple{names, NTuple{N_tuple, Data.Cell{numbertype, S}}}` | `Union{Data.CellTuple{N_tuple, numbertype, S}, Data.NamedCellTuple{N_tuple, numbertype, S}}`

--------------------------------------------------------------------------------
!!! note "Advanced"
        Data.DeviceArray{numbertype, ndims}

    The datatype `Data.DeviceArray` is automatically chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (Array for Threads, CUDA.CuDeviceArray for CUDA and AMDGPU.ROCDeviceArray for AMDGPU).

    !!! warning
        This datatype is not intended for explicit manual usage. [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray and AMDGPU.ROCArray automatically to CUDA.CuDeviceArray and AMDGPU.ROCDeviceArray in kernels when required.

    --------------------------------------------------------------------------------
        Data.DeviceCellArray{numbertype, ndims}

    The datatype `Data.DeviceCellArray` is automatically chosen to be compatible with the package for parallelization selected with [`@init_parallel_kernel`](@ref) (CPUCellArray for Threads, CuDeviceCellArray for CUDA and ROCDeviceCellArray for AMDGPU).

    !!! warning
        This datatype is not intended for explicit manual usage. [`@parallel`](@ref) and [`@parallel_indices`](@ref) convert CUDA.CuArray and AMDGPU.ROCArray automatically to CUDA.CuDeviceArray and AMDGPU.ROCDeviceArray in kernels when required.
"""

function Data_cuda(numbertype::DataType)
    if numbertype == NUMBERTYPE_NONE
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import Base, ParallelStencil.ParallelKernel.CUDA, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const Array{T, N}                   = CUDA.CuArray{T, N}
            const DeviceArray{T, N}             = CUDA.CuDeviceArray{T, N}
            const Cell{T, S}                    = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const DeviceCell{T, S}              = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const CellArray{T_elem, N, B}       = CellArrays.CuCellArray{<:Cell{T_elem},N,B,T_elem}
            const DeviceCellArray{T_elem, N, B} = CellArrays.CellArray{<:DeviceCell{T_elem},N,B,<:CUDA.CuDeviceArray{T_elem,CellArrays._N}}
        end)
    else
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import Base, ParallelStencil.ParallelKernel.CUDA, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const Number                         = $numbertype
            const Array{N}                       = CUDA.CuArray{$numbertype, N}
            const DeviceArray{N}                 = CUDA.CuDeviceArray{$numbertype, N}
            const Cell{S}                        = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            const DeviceCell{S}                  = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            const CellArray{N, B}                = CellArrays.CuCellArray{<:Cell,N,B,$numbertype}
            const DeviceCellArray{N, B}          = CellArrays.CellArray{<:DeviceCell,N,B,<:CUDA.CuDeviceArray{$numbertype,CellArrays._N}}
            const TArray{T, N}                   = CUDA.CuArray{T, N}
            const DeviceTArray{T, N}             = CUDA.CuDeviceArray{T, N}
            const TCell{T, S}                    = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const DeviceTCell{T, S}              = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const TCellArray{T_elem, N, B}       = CellArrays.CuCellArray{<:TCell{T_elem},N,B,T_elem}
            const DeviceTCellArray{T_elem, N, B} = CellArrays.CellArray{<:DeviceTCell{T_elem},N,B,<:CUDA.CuDeviceArray{T_elem,CellArrays._N}}
        end)
    end
end

function Data_amdgpu(numbertype::DataType)
    if numbertype == NUMBERTYPE_NONE
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import Base, ParallelStencil.ParallelKernel.AMDGPU, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const Array{T, N}                   = AMDGPU.ROCArray{T, N}
            const DeviceArray{T, N}             = AMDGPU.ROCDeviceArray{T, N}
            const Cell{T, S}                    = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const DeviceCell{T, S}              = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const CellArray{T_elem, N, B}       = CellArrays.ROCCellArray{<:Cell{T_elem},N,B,T_elem}
            const DeviceCellArray{T_elem, N, B} = CellArrays.CellArray{<:DeviceCell{T_elem},N,B,<:AMDGPU.ROCDeviceArray{T_elem,CellArrays._N}}
        end)
    else
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import Base, ParallelStencil.ParallelKernel.AMDGPU, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const Number                         = $numbertype
            const Array{N}                       = AMDGPU.ROCArray{$numbertype, N}
            const DeviceArray{N}                 = AMDGPU.ROCDeviceArray{$numbertype, N}
            const Cell{S}                        = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            const DeviceCell{S}                  = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            const CellArray{N, B}                = CellArrays.ROCCellArray{<:Cell,N,B,$numbertype}
            const DeviceCellArray{N, B}          = CellArrays.CellArray{<:DeviceCell,N,B,<:AMDGPU.ROCDeviceArray{$numbertype,CellArrays._N}}
            const TArray{T, N}                   = AMDGPU.ROCArray{T, N}
            const DeviceTArray{T, N}             = AMDGPU.ROCDeviceArray{T, N}
            const TCell{T, S}                    = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const DeviceTCell{T, S}              = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const TCellArray{T_elem, N, B}       = CellArrays.ROCCellArray{<:TCell{T_elem},N,B,T_elem}
            const DeviceTCellArray{T_elem, N, B} = CellArrays.CellArray{<:DeviceTCell{T_elem},N,B,<:AMDGPU.ROCDeviceArray{T_elem,CellArrays._N}}
        end)
    end
end

function Data_threads(numbertype::DataType)
    if numbertype == NUMBERTYPE_NONE
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import Base, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const Array{T, N}                    = Base.Array{T, N}
            const DeviceArray{T, N}              = Base.Array{T, N}
            const Cell{T, S}                     = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const DeviceCell{T, S}               = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const CellArray{T_elem, N, B}        = CellArrays.CPUCellArray{<:Cell{T_elem},N,B,T_elem}
            const DeviceCellArray{T_elem, N, B}  = CellArrays.CPUCellArray{<:DeviceCell{T_elem},N,B,T_elem}
        end)
    else
        :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
            import Base, ParallelStencil.ParallelKernel.CellArrays, ParallelStencil.ParallelKernel.StaticArrays
            const Number                         = $numbertype
            const Array{N}                       = Base.Array{$numbertype, N}
            const DeviceArray{N}                 = Base.Array{$numbertype, N}
            const Cell{S}                        = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            const DeviceCell{S}                  = Union{StaticArrays.SArray{S, $numbertype}, StaticArrays.FieldArray{S, $numbertype}}
            const CellArray{N, B}                = CellArrays.CPUCellArray{<:Cell,N,B,$numbertype}
            const DeviceCellArray{N, B}          = CellArrays.CPUCellArray{<:DeviceCell,N,B,$numbertype}
            const TArray{T, N}                   = Base.Array{T, N}
            const DeviceTArray{T, N}             = Base.Array{T, N}
            const TCell{T, S}                    = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const DeviceTCell{T, S}              = Union{StaticArrays.SArray{S, T}, StaticArrays.FieldArray{S, T}}
            const TCellArray{T_elem, N, B}       = CellArrays.CPUCellArray{<:TCell{T_elem},N,B,T_elem}
            const DeviceTCellArray{T_elem, N, B} = CellArrays.CPUCellArray{<:DeviceTCell{T_elem},N,B,T_elem}
        end)
    end
end

function Data_shared(numbertype::DataType)
    if numbertype == NUMBERTYPE_NONE
        quote
            const NumberTuple{N_tuple, T}                                  = NTuple{N_tuple, T}
            const ArrayTuple{N_tuple, T, N}                                = NTuple{N_tuple, Array{T, N}}
            const DeviceArrayTuple{N_tuple, T, N}                          = NTuple{N_tuple, DeviceArray{T, N}}
            const CellTuple{N_tuple, T, S}                                 = NTuple{N_tuple, Cell{T, S}}
            const DeviceCellTuple{N_tuple, T, S}                           = NTuple{N_tuple, DeviceCell{T, S}}
            const CellArrayTuple{N_tuple, T_elem, N, B}                    = NTuple{N_tuple, CellArray{T_elem, N, B}}
            const DeviceCellArrayTuple{N_tuple, T_elem, N, B}              = NTuple{N_tuple, DeviceCellArray{T_elem, N, B}}

            const NamedNumberTuple{N_tuple, T, names}                      = NamedTuple{names, <:NumberTuple{N_tuple, T}}
            const NamedArrayTuple{N_tuple, T, N, names}                    = NamedTuple{names, <:ArrayTuple{N_tuple, T, N}}
            const NamedDeviceArrayTuple{N_tuple, T, N, names}              = NamedTuple{names, <:DeviceArrayTuple{N_tuple, T, N}}
            const NamedCellTuple{N_tuple, T, S, names}                     = NamedTuple{names, <:CellTuple{N_tuple, T, S}}
            const NamedDeviceCellTuple{N_tuple, T, S, names}               = NamedTuple{names, <:DeviceCellTuple{N_tuple, T, S}}
            const NamedCellArrayTuple{N_tuple, T_elem, N, B, names}        = NamedTuple{names, <:CellArrayTuple{N_tuple, T_elem, N, B}}
            const NamedDeviceCellArrayTuple{N_tuple, T_elem, N, B, names}  = NamedTuple{names, <:DeviceCellArrayTuple{N_tuple, T_elem, N, B}}

            const NumberCollection{N_tuple, T}                             = Union{NumberTuple{N_tuple, T}, NamedNumberTuple{N_tuple, T}}
            const ArrayCollection{N_tuple, T, N}                           = Union{ArrayTuple{N_tuple, T, N}, NamedArrayTuple{N_tuple, T, N}}
            const DeviceArrayCollection{N_tuple, T, N}                     = Union{DeviceArrayTuple{N_tuple, T, N}, NamedDeviceArrayTuple{N_tuple, T, N}}
            const CellCollection{N_tuple, T, S}                            = Union{CellTuple{N_tuple, T, S}, NamedCellTuple{N_tuple, T, S}}
            const DeviceCellCollection{N_tuple, T, S}                      = Union{DeviceCellTuple{N_tuple, T, S}, NamedDeviceCellTuple{N_tuple, T, S}}
            const CellArrayCollection{N_tuple, T_elem, N, B}               = Union{CellArrayTuple{N_tuple, T_elem, N, B}, NamedCellArrayTuple{N_tuple, T_elem, N, B}}
            const DeviceCellArrayCollection{N_tuple, T_elem, N, B}         = Union{DeviceCellArrayTuple{N_tuple, T_elem, N, B}, NamedDeviceCellArrayTuple{N_tuple, T_elem, N, B}}

            NamedNumberTuple{}(T, t::NamedTuple)                     = Base.map(T, t)
            NamedArrayTuple{}(T, t::NamedTuple)                      = Base.map(Data.Array{T}, t)
            NamedCellTuple{}(T, t::NamedTuple)                       = Base.map(Data.Cell{T}, t)
            NamedCellArrayTuple{}(T, t::NamedTuple)                  = Base.map(Data.CellArray{T}, t)
        end
    else
        quote
            const NumberTuple{N_tuple}                                     = NTuple{N_tuple, Number}
            const ArrayTuple{N_tuple, N}                                   = NTuple{N_tuple, Array{N}}
            const DeviceArrayTuple{N_tuple, N}                             = NTuple{N_tuple, DeviceArray{N}}
            const CellTuple{N_tuple, S}                                    = NTuple{N_tuple, Cell{S}}
            const DeviceCellTuple{N_tuple, S}                              = NTuple{N_tuple, DeviceCell{S}}
            const CellArrayTuple{N_tuple, N, B}                            = NTuple{N_tuple, CellArray{N, B}}
            const DeviceCellArrayTuple{N_tuple, N, B}                      = NTuple{N_tuple, DeviceCellArray{N, B}}
            const TNumberTuple{N_tuple, T}                                 = NTuple{N_tuple, T}
            const TArrayTuple{N_tuple, T, N}                               = NTuple{N_tuple, TArray{T, N}}
            const DeviceTArrayTuple{N_tuple, T, N}                         = NTuple{N_tuple, DeviceTArray{T, N}}
            const TCellTuple{N_tuple, T, S}                                = NTuple{N_tuple, TCell{T, S}}
            const DeviceTCellTuple{N_tuple, T, S}                          = NTuple{N_tuple, DeviceTCell{T, S}}
            const TCellArrayTuple{N_tuple, T_elem, N, B}                   = NTuple{N_tuple, TCellArray{T_elem, N, B}}
            const DeviceTCellArrayTuple{N_tuple, T_elem, N, B}             = NTuple{N_tuple, DeviceTCellArray{T_elem, N, B}}

            const NamedNumberTuple{N_tuple, names}                         = NamedTuple{names, <:NumberTuple{N_tuple}}
            const NamedArrayTuple{N_tuple, N, names}                       = NamedTuple{names, <:ArrayTuple{N_tuple, N}}
            const NamedDeviceArrayTuple{N_tuple, N, names}                 = NamedTuple{names, <:DeviceArrayTuple{N_tuple, N}}
            const NamedCellTuple{N_tuple, S, names}                        = NamedTuple{names, <:CellTuple{N_tuple, S}}
            const NamedDeviceCellTuple{N_tuple, S, names}                  = NamedTuple{names, <:DeviceCellTuple{N_tuple, S}}
            const NamedCellArrayTuple{N_tuple, N, B, names}                = NamedTuple{names, <:CellArrayTuple{N_tuple, N, B}}
            const NamedDeviceCellArrayTuple{N_tuple, N, B, names}          = NamedTuple{names, <:DeviceCellArrayTuple{N_tuple, N, B}}
            const NamedTNumberTuple{N_tuple, T, names}                     = NamedTuple{names, <:TNumberTuple{N_tuple, T}}
            const NamedTArrayTuple{N_tuple, T, N, names}                   = NamedTuple{names, <:TArrayTuple{N_tuple, T, N}}
            const NamedDeviceTArrayTuple{N_tuple, T, N, names}             = NamedTuple{names, <:DeviceTArrayTuple{N_tuple, T, N}}
            const NamedTCellTuple{N_tuple, T, S, names}                    = NamedTuple{names, <:TCellTuple{N_tuple, T, S}}
            const NamedDeviceTCellTuple{N_tuple, T, S, names}              = NamedTuple{names, <:DeviceTCellTuple{N_tuple, T, S}}
            const NamedTCellArrayTuple{N_tuple, T_elem, N, B, names}       = NamedTuple{names, <:TCellArrayTuple{N_tuple, T_elem, N, B}}
            const NamedDeviceTCellArrayTuple{N_tuple, T_elem, N, B, names} = NamedTuple{names, <:DeviceTCellArrayTuple{N_tuple, T_elem, N, B}}

            const NumberCollection{N_tuple}                                = Union{NumberTuple{N_tuple}, NamedNumberTuple{N_tuple}}
            const ArrayCollection{N_tuple, N}                              = Union{ArrayTuple{N_tuple, N}, NamedArrayTuple{N_tuple, N}}
            const DeviceArrayCollection{N_tuple, N}                        = Union{DeviceArrayTuple{N_tuple, N}, NamedDeviceArrayTuple{N_tuple, N}}
            const CellCollection{N_tuple, S}                               = Union{CellTuple{N_tuple, S}, NamedCellTuple{N_tuple, S}}
            const DeviceCellCollection{N_tuple, S}                         = Union{DeviceCellTuple{N_tuple, S}, NamedDeviceCellTuple{N_tuple, S}}
            const CellArrayCollection{N_tuple, N, B}                       = Union{CellArrayTuple{N_tuple, N, B}, NamedCellArrayTuple{N_tuple, N, B}}
            const DeviceCellArrayCollection{N_tuple, N, B}                 = Union{DeviceCellArrayTuple{N_tuple, N, B}, NamedDeviceCellArrayTuple{N_tuple, N, B}}
            const TNumberCollection{N_tuple, T}                            = Union{TNumberTuple{N_tuple, T}, NamedTNumberTuple{N_tuple, T}}
            const TArrayCollection{N_tuple, T, N}                          = Union{TArrayTuple{N_tuple, T, N}, NamedTArrayTuple{N_tuple, T, N}}
            const DeviceTArrayCollection{N_tuple, T, N}                    = Union{DeviceTArrayTuple{N_tuple, T, N}, NamedDeviceTArrayTuple{N_tuple, T, N}}
            const TCellCollection{N_tuple, T, S}                           = Union{TCellTuple{N_tuple, T, S}, NamedTCellTuple{N_tuple, T, S}}
            const DeviceTCellCollection{N_tuple, T, S}                     = Union{DeviceTCellTuple{N_tuple, T, S}, NamedDeviceTCellTuple{N_tuple, T, S}}
            const TCellArrayCollection{N_tuple, T_elem, N, B}              = Union{TCellArrayTuple{N_tuple, T_elem, N, B}, NamedTCellArrayTuple{N_tuple, T_elem, N, B}}
            const DeviceTCellArrayCollection{N_tuple, T_elem, N, B}        = Union{DeviceTCellArrayTuple{N_tuple, T_elem, N, B}, NamedDeviceTCellArrayTuple{N_tuple, T_elem, N, B}}

            NamedNumberTuple{}(t::NamedTuple)                        = Base.map(Data.Number, t)
            NamedArrayTuple{}(t::NamedTuple)                         = Base.map(Data.Array, t)
            NamedCellTuple{}(t::NamedTuple)                          = Base.map(Data.Cell, t)
            NamedCellArrayTuple{}(t::NamedTuple)                     = Base.map(Data.CellArray, t)
            NamedTNumberTuple{}(T, t::NamedTuple)                    = Base.map(T, t)
            NamedTArrayTuple{}(T, t::NamedTuple)                     = Base.map(Data.TArray{T}, t)
            NamedTCellTuple{}(T, t::NamedTuple)                      = Base.map(Data.TCell{T}, t)
            NamedTCellArrayTuple{}(T, t::NamedTuple)                 = Base.map(Data.TCellArray{T}, t)
        end
    end
end

function Data_none()
    :(baremodule Data # NOTE: there cannot be any newline before 'module Data' or it will create a begin end block and the module creation will fail.
    end)
end
