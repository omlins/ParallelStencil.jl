"""
Module FiniteDifferences1D

Provides macros for 1-D finite differences computations. The dimensions x refers to the 1st and only array dimension. The module was designed to enable a math-close notation and, as a consequence, it does not allow for a nested invocation of the provided macros as, e.g., `@d(@d(A))` (instead, one can simply write `@d2(A)`). Additional macros can be straightforwardly added in the caller module to cover eventual missing functionality. The module is intended for usage with the module ParallelStencil.

# Usage
    using ParallelStencil.FiniteDifferences1D

# Macros

###### Differences
- [`@d`](@ref)
- [`@d2`](@ref)

###### Selection
- [`@all`](@ref)
- [`@inn`](@ref)

###### Averages
- [`@av`](@ref)

###### Harmonic averages
- [`@harm`](@ref)

###### Others
- [`@maxloc`](@ref)
- [`@minloc`](@ref)

To see a description of a macro type `?<macroname>` (including the `@`).
"""
module FiniteDifferences1D
export @d, @d2
export @all, @inn
export @av
export @harm
export @maxloc, @minloc
export @within

@doc "`@d(A)`: Compute differences between adjacent elements of `A`." :(@d)
@doc "`@d2(A)`: Compute the 2nd order differences between adjacent elements of `A`." :(@d2)
@doc "`@all(A)`: Select all elements of `A`. Corresponds to `A[:]`." :(@all)
@doc "`@inn(A)`: Select the inner elements of `A`. Corresponds to `A[2:end-1]`" :(@inn)
@doc "`@av(A)`: Compute averages between adjacent elements of `A`." :(@av)
@doc "`@harm(A)`: Compute harmonic averages between adjacent elements of `A`." :(@harm)
@doc "`@maxloc(A)`: Compute the maximum between 2nd order adjacent elements of `A`, using a moving window of size 3." :(@maxloc)
@doc "`@minloc(A)`: Compute the minimum between 2nd order adjacent elements of `A`, using a moving window of size 3." :(@minloc)

import ..ParallelStencil
import ..ParallelStencil: INDICES, WITHIN_DOC, @expandargs
const ix = INDICES[1]
const ixi = :($ix+1)

macro      d(A)  @expandargs(A);  esc(:( $A[$ix+1] - $A[$ix] )) end
macro     d2(A)  @expandargs(A);  esc(:( ($A[$ixi+1] - $A[$ixi])  -  ($A[$ixi] - $A[$ixi-1]) )) end
macro    all(A)  @expandargs(A);  esc(:( $A[$ix  ] )) end
macro    inn(A)  @expandargs(A);  esc(:( $A[$ixi ] )) end
macro     av(A)  @expandargs(A);  esc(:(($A[$ix] + $A[$ix+1] )/2 )) end
macro   harm(A)  @expandargs(A);  esc(:(1/(1/$A[$ix] + 1/$A[$ix+1])*2 )) end
macro maxloc(A)  @expandargs(A);  esc(:( max( max($A[$ixi-1], $A[$ixi+1]), $A[$ixi] ) )) end
macro minloc(A)  @expandargs(A);  esc(:( min( min($A[$ixi-1], $A[$ixi+1]), $A[$ixi] ) )) end

@doc WITHIN_DOC
macro within(macroname::String, A)
    @expandargs(A)
    if     macroname == "@all"  esc(  :($ix<=size($A,1)  )  )
    elseif macroname == "@inn"  esc(  :($ix<=size($A,1)-2)  )
    else error("unkown macroname: $macroname. If you want to add your own assignement macros, overwrite the macro 'within(macroname::String, A)'; to still use the exising macro within as well call ParallelStencil.FiniteDifferences{1|2|3}D.@within(macroname, A) at the end.")
    end
end

end # Module FiniteDifferences1D


"""
Module FiniteDifferences2D

Provides macros for 2-D finite differences computations. The dimensions x and y refer to the 1st and 2nd array dimensions, respectively. The module was designed to enable a math-close notation and, as a consequence, it does not allow for a nested invocation of the provided macros as, e.g., `@inn_y(@d_xa(A))` (instead, one can simply write `@d_xi(A)`). Additional macros can be straightforwardly added in the caller module to cover eventual missing functionality. The module is intended for usage with the module ParallelStencil.

# Usage
    using ParallelStencil.FiniteDifferences2D

# Macros

###### Differences
- [`@d_xa`](@ref)
- [`@d_ya`](@ref)
- [`@d_xi`](@ref)
- [`@d_yi`](@ref)
- [`@d2_xa`](@ref)
- [`@d2_ya`](@ref)
- [`@d2_xi`](@ref)
- [`@d2_yi`](@ref)

###### Selection
- [`@all`](@ref)
- [`@inn`](@ref)
- [`@inn_x`](@ref)
- [`@inn_y`](@ref)

###### Averages
- [`@av`](@ref)
- [`@av_xa`](@ref)
- [`@av_ya`](@ref)
- [`@av_xi`](@ref)
- [`@av_yi`](@ref)

###### Harmonic averages
- [`@harm`](@ref)
- [`@harm_xa`](@ref)
- [`@harm_ya`](@ref)
- [`@harm_xi`](@ref)
- [`@harm_yi`](@ref)

###### Others
- [`@maxloc`](@ref)
- [`@minloc`](@ref)

To see a description of a macro type `?<macroname>` (including the `@`).
"""
module FiniteDifferences2D
export @d_xa, @d_ya, @d_xi, @d_yi, @d2_xa, @d2_ya, @d2_xi, @d2_yi
export @all, @inn, @inn_x, @inn_y
export @av, @av_xa, @av_ya, @av_xi, @av_yi
export @harm, @harm_xa, @harm_ya, @harm_xi, @harm_yi
export @maxloc, @minloc
export @within

@doc "`@d_xa(A)`: Compute differences between adjacent elements of `A` along the dimension x." :(@d_xa)
@doc "`@d_ya(A)`: Compute differences between adjacent elements of `A` along the dimension y." :(@d_ya)
@doc "`@d_xi(A)`: Compute differences between adjacent elements of `A` along the dimension x and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_y(@d_xa(A))`." :(@d_xi)
@doc "`@d_yi(A)`: Compute differences between adjacent elements of `A` along the dimension y and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_x(@d_ya(A))`." :(@d_yi)
@doc "`@d2_xa(A)`: Compute the 2nd order differences between adjacent elements of `A` along the dimension x." :(@d2_xa)
@doc "`@d2_ya(A)`: Compute the 2nd order differences between adjacent elements of `A` along the dimension y." :(@d2_ya)
@doc "`@d2_xi(A)`: Compute the 2nd order differences between adjacent elements of `A` along the dimension x and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_y(@d2_xa(A))`." :(@d2_xi)
@doc "`@d2_yi(A)`: Compute the 2nd order differences between adjacent elements of `A` along the dimension y and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_x(@d2_ya(A))`." :(@d2_yi)
@doc "`@all(A)`: Select all elements of `A`. Corresponds to `A[:,:]`." :(@all)
@doc "`@inn(A)`: Select the inner elements of `A`. Corresponds to `A[2:end-1,2:end-1]`." :(@inn)
@doc "`@inn_x(A)`: Select the inner elements of `A` in dimension x. Corresponds to `A[2:end-1,:]`." :(@inn_x)
@doc "`@inn_y(A)`: Select the inner elements of `A` in dimension y. Corresponds to `A[:,2:end-1]`." :(@inn_y)
@doc "`@av(A)`: Compute averages between adjacent elements of `A` along the dimensions x and y." :(@av)
@doc "`@av_xa(A)`: Compute averages between adjacent elements of `A` along the dimension x." :(@av_xa)
@doc "`@av_ya(A)`: Compute averages between adjacent elements of `A` along the dimension y." :(@av_ya)
@doc "`@av_xi(A)`: Compute averages between adjacent elements of `A` along the dimension x and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_y(@av_xa(A))`." :(@av_xi)
@doc "`@av_yi(A)`: Compute averages between adjacent elements of `A` along the dimension y and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_x(@av_ya(A))`." :(@av_yi)
@doc "`@harm(A)`: Compute harmonic averages between adjacent elements of `A` along the dimensions x and y." :(@harm)
@doc "`@harm_xa(A)`: Compute harmonic averages between adjacent elements of `A` along the dimension x." :(@harm_xa)
@doc "`@harm_ya(A)`: Compute harmonic averages between adjacent elements of `A` along the dimension y." :(@harm_ya)
@doc "`@harm_xi(A)`: Compute harmonic averages between adjacent elements of `A` along the dimension x and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_y(@harm_xa(A))`." :(@harm_xi)
@doc "`@harm_yi(A)`: Compute harmonic averages between adjacent elements of `A` along the dimension y and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_x(@harm_ya(A))`." :(@harm_yi)
@doc "`@maxloc(A)`: Compute the maximum between 2nd order adjacent elements of `A`, using a moving window of size 3." :(@maxloc)
@doc "`@minloc(A)`: Compute the minimum between 2nd order adjacent elements of `A`, using a moving window of size 3." :(@minloc)

import ..ParallelStencil
import ..ParallelStencil: INDICES, WITHIN_DOC, @expandargs
ix, iy = INDICES[1], INDICES[2]
ixi, iyi = :($ix+1), :($iy+1)

macro     d_xa(A)  @expandargs(A);  esc(:( $A[$ix+1,$iy  ] - $A[$ix  ,$iy ] )) end
macro     d_ya(A)  @expandargs(A);  esc(:( $A[$ix  ,$iy+1] - $A[$ix  ,$iy ] )) end
macro     d_xi(A)  @expandargs(A);  esc(:( $A[$ix+1,$iyi ] - $A[$ix  ,$iyi] )) end
macro     d_yi(A)  @expandargs(A);  esc(:( $A[$ixi ,$iy+1] - $A[$ixi ,$iy ] )) end
macro    d2_xa(A)  @expandargs(A);  esc(:( ($A[$ixi+1,$iy   ] - $A[$ixi ,$iy ])  -  ($A[$ixi ,$iy ] - $A[$ixi-1,$iy   ]) )) end
macro    d2_ya(A)  @expandargs(A);  esc(:( ($A[$ix   ,$iyi+1] - $A[$ix  ,$iyi])  -  ($A[$ix  ,$iyi] - $A[$ix   ,$iyi-1]) )) end
macro    d2_xi(A)  @expandargs(A);  esc(:( ($A[$ixi+1,$iyi  ] - $A[$ixi ,$iyi])  -  ($A[$ixi ,$iyi] - $A[$ixi-1,$iyi  ]) )) end
macro    d2_yi(A)  @expandargs(A);  esc(:( ($A[$ixi  ,$iyi+1] - $A[$ixi ,$iyi])  -  ($A[$ixi ,$iyi] - $A[$ixi  ,$iyi-1]) )) end
macro      all(A)  @expandargs(A);  esc(:( $A[$ix  ,$iy  ] )) end
macro      inn(A)  @expandargs(A);  esc(:( $A[$ixi ,$iyi ] )) end
macro    inn_x(A)  @expandargs(A);  esc(:( $A[$ixi ,$iy  ] )) end
macro    inn_y(A)  @expandargs(A);  esc(:( $A[$ix  ,$iyi ] )) end
macro       av(A)  @expandargs(A);  esc(:(($A[$ix  ,$iy  ] + $A[$ix+1,$iy  ] + $A[$ix,$iy+1] + $A[$ix+1,$iy+1])/4 )) end
macro    av_xa(A)  @expandargs(A);  esc(:(($A[$ix  ,$iy  ] + $A[$ix+1,$iy  ] )/2 )) end
macro    av_ya(A)  @expandargs(A);  esc(:(($A[$ix  ,$iy  ] + $A[$ix  ,$iy+1] )/2 )) end
macro    av_xi(A)  @expandargs(A);  esc(:(($A[$ix  ,$iyi ] + $A[$ix+1,$iyi ] )/2 )) end
macro    av_yi(A)  @expandargs(A);  esc(:(($A[$ixi ,$iy  ] + $A[$ixi ,$iy+1] )/2 )) end
macro     harm(A)  @expandargs(A);  esc(:(1/(1/$A[$ix  ,$iy  ] + 1/$A[$ix+1,$iy  ] + 1/$A[$ix,$iy+1] + 1/$A[$ix+1,$iy+1])*4 )) end
macro  harm_xa(A)  @expandargs(A);  esc(:(1/(1/$A[$ix  ,$iy  ] + 1/$A[$ix+1,$iy  ] )*2 )) end
macro  harm_ya(A)  @expandargs(A);  esc(:(1/(1/$A[$ix  ,$iy  ] + 1/$A[$ix  ,$iy+1] )*2 )) end
macro  harm_xi(A)  @expandargs(A);  esc(:(1/(1/$A[$ix  ,$iyi ] + 1/$A[$ix+1,$iyi ] )*2 )) end
macro  harm_yi(A)  @expandargs(A);  esc(:(1/(1/$A[$ixi ,$iy  ] + 1/$A[$ixi ,$iy+1] )*2 )) end
macro   maxloc(A)  @expandargs(A);  esc(:( max( max( max($A[$ixi-1,$iyi  ], $A[$ixi+1,$iyi  ])  , $A[$ixi  ,$iyi  ] ),
                                                     max($A[$ixi  ,$iyi-1], $A[$ixi  ,$iyi+1]) ) )) end
macro   minloc(A)  @expandargs(A);  esc(:( min( min( min($A[$ixi-1,$iyi  ], $A[$ixi+1,$iyi  ])  , $A[$ixi  ,$iyi  ] ),
                                                     min($A[$ixi  ,$iyi-1], $A[$ixi  ,$iyi+1]) ) )) end

@doc WITHIN_DOC
macro within(macroname::String, A)
    @expandargs(A)
    if     macroname == "@all"    esc(  :($ix<=size($A,1)   && $iy<=size($A,2)  )  )
    elseif macroname == "@inn"    esc(  :($ix<=size($A,1)-2 && $iy<=size($A,2)-2)  )
    elseif macroname == "@inn_x"  esc(  :($ix<=size($A,1)-2 && $iy<=size($A,2)  )  )
    elseif macroname == "@inn_y"  esc(  :($ix<=size($A,1)   && $iy<=size($A,2)-2)  )
    else error("unkown macroname: $macroname. If you want to add your own assignement macros, overwrite the macro 'within(macroname::String, A)'; to still use the exising macro within as well call ParallelStencil.FiniteDifferences{1|2|3}D.@within(macroname, A) at the end.")
    end
end

end # Module FiniteDifferences2D


"""
Module FiniteDifferences3D

Provides macros for 3-D finite differences computations. The dimensions x, y and z refer to the 1st, 2nd and 3rd array dimensions, respectively. The module was designed to enable a math-close notation and, as a consequence, it does not allow for a nested invocation of the provided macros as, e.g., `@inn_z(@inn_y(@d_xa(A)))` (instead, one can simply write `@d_xi(A)`). Additional macros can be straightforwardly added in the caller module to cover eventual missing functionality. The module is intended for usage with the module ParallelStencil.

# Usage
    using ParallelStencil.FiniteDifferences3D

# Macros

###### Differences
- [`@d_xa`](@ref)
- [`@d_ya`](@ref)
- [`@d_za`](@ref)
- [`@d_xi`](@ref)
- [`@d_yi`](@ref)
- [`@d_zi`](@ref)
- [`@d2_xi`](@ref)
- [`@d2_yi`](@ref)
- [`@d2_zi`](@ref)

###### Selection
- [`@all`](@ref)
- [`@inn`](@ref)
- [`@inn_x`](@ref)
- [`@inn_y`](@ref)
- [`@inn_z`](@ref)
- [`@inn_xy`](@ref)
- [`@inn_xz`](@ref)
- [`@inn_yz`](@ref)

###### Averages
- [`@av`](@ref)
- [`@av_xa`](@ref)
- [`@av_ya`](@ref)
- [`@av_za`](@ref)
- [`@av_xi`](@ref)
- [`@av_yi`](@ref)
- [`@av_zi`](@ref)
- [`@av_xya`](@ref)
- [`@av_xza`](@ref)
- [`@av_yza`](@ref)
- [`@av_xyi`](@ref)
- [`@av_xzi`](@ref)
- [`@av_yzi`](@ref)

###### Harmonic averages
- [`@harm`](@ref)
- [`@harm_xa`](@ref)
- [`@harm_ya`](@ref)
- [`@harm_za`](@ref)
- [`@harm_xi`](@ref)
- [`@harm_yi`](@ref)
- [`@harm_zi`](@ref)
- [`@harm_xya`](@ref)
- [`@harm_xza`](@ref)
- [`@harm_yza`](@ref)
- [`@harm_xyi`](@ref)
- [`@harm_xzi`](@ref)
- [`@harm_yzi`](@ref)

###### Others
- [`@maxloc`](@ref)
- [`@minloc`](@ref)

To see a description of a macro type `?<macroname>` (including the `@`).
"""
module FiniteDifferences3D
export @d_xa, @d_ya, @d_za, @d_xi, @d_yi, @d_zi, @d2_xi, @d2_yi, @d2_zi
export @all, @inn, @inn_x, @inn_y, @inn_z, @inn_xy, @inn_xz, @inn_yz
export @av, @av_xa, @av_ya, @av_za, @av_xi, @av_yi, @av_zi, @av_xya, @av_xza, @av_yza, @av_xyi, @av_xzi, @av_yzi #, @av_xya2, @av_xza2, @av_yza2
export @harm, @harm_xa, @harm_ya, @harm_za, @harm_xi, @harm_yi, @harm_zi, @harm_xya, @harm_xza, @harm_yza, @harm_xyi, @harm_xzi, @harm_yzi 
export @maxloc, @minloc
export @within

@doc "`@d_xa(A)`: Compute differences between adjacent elements of `A` along the dimension x." :(@d_xa)
@doc "`@d_ya(A)`: Compute differences between adjacent elements of `A` along the dimension y." :(@d_ya)
@doc "`@d_za(A)`: Compute differences between adjacent elements of `A` along the dimension z." :(@d_za)
@doc "`@d_xi(A)`: Compute differences between adjacent elements of `A` along the dimension x and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_yz(@d_xa(A))`." :(@d_xi)
@doc "`@d_yi(A)`: Compute differences between adjacent elements of `A` along the dimension y and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_xz(@d_ya(A))`." :(@d_yi)
@doc "`@d_zi(A)`: Compute differences between adjacent elements of `A` along the dimension z and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_xy(@d_za(A))`." :(@d_zi)
@doc "`@d2_xi(A)`: Compute the 2nd order differences between adjacent elements of `A` along the dimension x and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_yz(@d2_xa(A))`." :(@d2_xi)
@doc "`@d2_yi(A)`: Compute the 2nd order differences between adjacent elements of `A` along the dimension y and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_xz(@d2_ya(A))`." :(@d2_yi)
@doc "`@d2_zi(A)`: Compute the 2nd order differences between adjacent elements of `A` along the dimension y and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_xy(@d2_za(A))`." :(@d2_zi)
@doc "`@all(A)`: Select all elements of `A`. Corresponds to `A[:,:,:]`." :(@all)
@doc "`@inn(A)`: Select the inner elements of `A`. Corresponds to `A[2:end-1,2:end-1,2:end-1]`." :(@inn)
@doc "`@inn_x(A)`: Select the inner elements of `A` in dimension x. Corresponds to `A[2:end-1,:,:]`." :(@inn_x)
@doc "`@inn_y(A)`: Select the inner elements of `A` in dimension y. Corresponds to `A[:,2:end-1,:]`." :(@inn_y)
@doc "`@inn_z(A)`: Select the inner elements of `A` in dimension z. Corresponds to `A[:,:,2:end-1]`." :(@inn_z)
@doc "`@inn_xy(A)`: Select the inner elements of `A` in dimensions x and y. Corresponds to `A[2:end-1,2:end-1,:]`." :(@inn_xy)
@doc "`@inn_xz(A)`: Select the inner elements of `A` in dimensions x and z. Corresponds to `A[2:end-1,:,2:end-1]`." :(@inn_xz)
@doc "`@inn_yz(A)`: Select the inner elements of `A` in dimensions y and z. Corresponds to `A[:,2:end-1,2:end-1]`." :(@inn_yz)
@doc "`@av(A)`: Compute averages between adjacent elements of `A` along the dimensions x and y and z." :(@av)
@doc "`@av_xa(A)`: Compute averages between adjacent elements of `A` along the dimension x." :(@av_xa)
@doc "`@av_ya(A)`: Compute averages between adjacent elements of `A` along the dimension y." :(@av_ya)
@doc "`@av_za(A)`: Compute averages between adjacent elements of `A` along the dimension z." :(@av_za)
@doc "`@av_xi(A)`: Compute averages between adjacent elements of `A` along the dimension x and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_yz(@av_xa(A))`." :(@av_xi)
@doc "`@av_yi(A)`: Compute averages between adjacent elements of `A` along the dimension y and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_xz(@av_ya(A))`." :(@av_yi)
@doc "`@av_zi(A)`: Compute averages between adjacent elements of `A` along the dimension z and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_xy(@av_za(A))`." :(@av_zi)
@doc "`@av_xya(A)`: Compute averages between adjacent elements of `A` along the dimensions x and y." :(@av_xya)
@doc "`@av_xza(A)`: Compute averages between adjacent elements of `A` along the dimensions x and z." :(@av_xza)
@doc "`@av_yza(A)`: Compute averages between adjacent elements of `A` along the dimensions y and z." :(@av_yza)
@doc "`@av_xyi(A)`: Compute averages between adjacent elements of `A` along the dimensions x and y and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_z(@av_xya(A))`." :(@av_xyi)
@doc "`@av_xzi(A)`: Compute averages between adjacent elements of `A` along the dimensions x and z and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_y(@av_xza(A))`." :(@av_xzi)
@doc "`@av_yzi(A)`: Compute averages between adjacent elements of `A` along the dimensions y and z and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_x(@av_yza(A))`." :(@av_yzi)
@doc "`@harm(A)`: Compute harmonic averages between adjacent elements of `A` along the dimensions x and y and z." :(@harm)
@doc "`@harm_xa(A)`: Compute harmonic averages between adjacent elements of `A` along the dimension x." :(@harm_xa)
@doc "`@harm_ya(A)`: Compute harmonic averages between adjacent elements of `A` along the dimension y." :(@harm_ya)
@doc "`@harm_za(A)`: Compute harmonic averages between adjacent elements of `A` along the dimension z." :(@harm_za)
@doc "`@harm_xi(A)`: Compute harmonic averages between adjacent elements of `A` along the dimension x and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_yz(@harm_xa(A))`." :(@harm_xi)
@doc "`@harm_yi(A)`: Compute harmonic averages between adjacent elements of `A` along the dimension y and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_xz(@harm_ya(A))`." :(@harm_yi)
@doc "`@harm_zi(A)`: Compute harmonic averages between adjacent elements of `A` along the dimension z and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_xy(@harm_za(A))`." :(@harm_zi)
@doc "`@harm_xya(A)`: Compute harmonic averages between adjacent elements of `A` along the dimensions x and y." :(@harm_xya)
@doc "`@harm_xza(A)`: Compute harmonic averages between adjacent elements of `A` along the dimensions x and z." :(@harm_xza)
@doc "`@harm_yza(A)`: Compute harmonic averages between adjacent elements of `A` along the dimensions y and z." :(@harm_yza)
@doc "`@harm_xyi(A)`: Compute harmonic averages between adjacent elements of `A` along the dimensions x and y and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_z(@harm_xya(A))`." :(@harm_xyi)
@doc "`@harm_xzi(A)`: Compute harmonic averages between adjacent elements of `A` along the dimensions x and z and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_y(@harm_xza(A))`." :(@harm_xzi)
@doc "`@harm_yzi(A)`: Compute harmonic averages between adjacent elements of `A` along the dimensions y and z and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_x(@harm_yza(A))`." :(@harm_yzi)
@doc "`@maxloc(A)`: Compute the maximum between 2nd order adjacent elements of `A`, using a moving window of size 3." :(@maxloc)
@doc "`@minloc(A)`: Compute the minimum between 2nd order adjacent elements of `A`, using a moving window of size 3." :(@minloc)

import ..ParallelStencil
import ..ParallelStencil: INDICES, WITHIN_DOC, @expandargs
ix, iy, iz = INDICES[1], INDICES[2], INDICES[3]
ixi, iyi, izi = :($ix+1), :($iy+1), :($iz+1)

macro     d_xa(A)  @expandargs(A);  esc(:( $A[$ix+1,$iy  ,$iz  ] - $A[$ix  ,$iy  ,$iz  ] )) end
macro     d_ya(A)  @expandargs(A);  esc(:( $A[$ix  ,$iy+1,$iz  ] - $A[$ix  ,$iy  ,$iz  ] )) end
macro     d_za(A)  @expandargs(A);  esc(:( $A[$ix  ,$iy  ,$iz+1] - $A[$ix  ,$iy  ,$iz  ] )) end
macro     d_xi(A)  @expandargs(A);  esc(:( $A[$ix+1,$iyi ,$izi ] - $A[$ix  ,$iyi ,$izi ] )) end
macro     d_yi(A)  @expandargs(A);  esc(:( $A[$ixi ,$iy+1,$izi ] - $A[$ixi ,$iy  ,$izi ] )) end
macro     d_zi(A)  @expandargs(A);  esc(:( $A[$ixi ,$iyi ,$iz+1] - $A[$ixi ,$iyi ,$iz  ] )) end
macro    d2_xi(A)  @expandargs(A);  esc(:( ($A[$ixi+1,$iyi  ,$izi  ] - $A[$ixi ,$iyi ,$izi ])  -  ($A[$ixi ,$iyi ,$izi ] - $A[$ixi-1,$iyi  ,$izi  ]) )) end
macro    d2_yi(A)  @expandargs(A);  esc(:( ($A[$ixi  ,$iyi+1,$izi  ] - $A[$ixi ,$iyi ,$izi ])  -  ($A[$ixi ,$iyi ,$izi ] - $A[$ixi  ,$iyi-1,$izi  ]) )) end
macro    d2_zi(A)  @expandargs(A);  esc(:( ($A[$ixi  ,$iyi  ,$izi+1] - $A[$ixi ,$iyi ,$izi ])  -  ($A[$ixi ,$iyi ,$izi ] - $A[$ixi  ,$iyi  ,$izi-1]) )) end
macro      all(A)  @expandargs(A);  esc(:( $A[$ix  ,$iy  ,$iz  ] )) end
macro      inn(A)  @expandargs(A);  esc(:( $A[$ixi ,$iyi ,$izi ] )) end
macro    inn_x(A)  @expandargs(A);  esc(:( $A[$ixi ,$iy  ,$iz  ] )) end
macro    inn_y(A)  @expandargs(A);  esc(:( $A[$ix  ,$iyi ,$iz  ] )) end
macro    inn_z(A)  @expandargs(A);  esc(:( $A[$ix  ,$iy  ,$izi ] )) end
macro   inn_xy(A)  @expandargs(A);  esc(:( $A[$ixi ,$iyi ,$iz  ] )) end
macro   inn_xz(A)  @expandargs(A);  esc(:( $A[$ixi ,$iy  ,$izi ] )) end
macro   inn_yz(A)  @expandargs(A);  esc(:( $A[$ix  ,$iyi ,$izi ] )) end
macro       av(A)  @expandargs(A);  esc(:(($A[$ix  ,$iy  ,$iz  ] + $A[$ix+1,$iy  ,$iz  ] +
                                           $A[$ix+1,$iy+1,$iz  ] + $A[$ix+1,$iy+1,$iz+1] +
                                           $A[$ix  ,$iy+1,$iz+1] + $A[$ix  ,$iy  ,$iz+1] +
                                           $A[$ix+1,$iy  ,$iz+1] + $A[$ix  ,$iy+1,$iz  ] )/8)) end
macro    av_xa(A)  @expandargs(A);  esc(:(($A[$ix  ,$iy  ,$iz  ] + $A[$ix+1,$iy  ,$iz  ] )/2 )) end
macro    av_ya(A)  @expandargs(A);  esc(:(($A[$ix  ,$iy  ,$iz  ] + $A[$ix  ,$iy+1,$iz  ] )/2 )) end
macro    av_za(A)  @expandargs(A);  esc(:(($A[$ix  ,$iy  ,$iz  ] + $A[$ix  ,$iy  ,$iz+1] )/2 )) end
macro    av_xi(A)  @expandargs(A);  esc(:(($A[$ix  ,$iyi ,$izi ] + $A[$ix+1,$iyi ,$izi ] )/2 )) end
macro    av_yi(A)  @expandargs(A);  esc(:(($A[$ixi ,$iy  ,$izi ] + $A[$ixi ,$iy+1,$izi ] )/2 )) end
macro    av_zi(A)  @expandargs(A);  esc(:(($A[$ixi ,$iyi ,$iz  ] + $A[$ixi ,$iyi ,$iz+1] )/2 )) end
macro   av_xya(A)  @expandargs(A);  esc(:(($A[$ix  ,$iy  ,$iz  ] + $A[$ix+1,$iy  ,$iz  ] +
                                                         $A[$ix  ,$iy+1,$iz  ] + $A[$ix+1,$iy+1,$iz  ] )/4 )) end
macro   av_xza(A)  @expandargs(A);  esc(:(($A[$ix  ,$iy  ,$iz  ] + $A[$ix+1,$iy  ,$iz  ] +
                                                         $A[$ix  ,$iy  ,$iz+1] + $A[$ix+1,$iy  ,$iz+1] )/4 )) end
macro   av_yza(A)  @expandargs(A);  esc(:(($A[$ix  ,$iy  ,$iz  ] + $A[$ix  ,$iy+1,$iz  ] +
                                                         $A[$ix  ,$iy  ,$iz+1] + $A[$ix  ,$iy+1,$iz+1] )/4 )) end
macro   av_xyi(A)  @expandargs(A);  esc(:(($A[$ix  ,$iy  ,$izi ] + $A[$ix+1,$iy  ,$izi ] +
                                                         $A[$ix  ,$iy+1,$izi ] + $A[$ix+1,$iy+1,$izi ] )/4 )) end
macro   av_xzi(A)  @expandargs(A);  esc(:(($A[$ix  ,$iyi ,$iz  ] + $A[$ix+1,$iyi ,$iz  ] +
                                                         $A[$ix  ,$iyi ,$iz+1] + $A[$ix+1,$iyi ,$iz+1] )/4 )) end
macro   av_yzi(A)  @expandargs(A);  esc(:(($A[$ixi ,$iy  ,$iz  ] + $A[$ixi ,$iy+1,$iz  ] +
                                                         $A[$ixi ,$iy  ,$iz+1] + $A[$ixi ,$iy+1,$iz+1] )/4 )) end
macro     harm(A)  @expandargs(A);  esc(:(1/(1/$A[$ix  ,$iy  ,$iz  ] + 1/$A[$ix+1,$iy  ,$iz  ] +
                                               1/$A[$ix+1,$iy+1,$iz  ] + 1/$A[$ix+1,$iy+1,$iz+1] +
                                               1/$A[$ix  ,$iy+1,$iz+1] + 1/$A[$ix  ,$iy  ,$iz+1] +
                                               1/$A[$ix+1,$iy  ,$iz+1] + 1/$A[$ix  ,$iy+1,$iz  ] )*8)) end
macro  harm_xa(A)  @expandargs(A);  esc(:(1/(1/$A[$ix  ,$iy  ,$iz  ] + 1/$A[$ix+1,$iy  ,$iz  ] )*2 )) end
macro  harm_ya(A)  @expandargs(A);  esc(:(1/(1/$A[$ix  ,$iy  ,$iz  ] + 1/$A[$ix  ,$iy+1,$iz  ] )*2 )) end
macro  harm_za(A)  @expandargs(A);  esc(:(1/(1/$A[$ix  ,$iy  ,$iz  ] + 1/$A[$ix  ,$iy  ,$iz+1] )*2 )) end
macro  harm_xi(A)  @expandargs(A);  esc(:(1/(1/$A[$ix  ,$iyi ,$izi ] + 1/$A[$ix+1,$iyi ,$izi ] )*2 )) end
macro  harm_yi(A)  @expandargs(A);  esc(:(1/(1/$A[$ixi ,$iy  ,$izi ] + 1/$A[$ixi ,$iy+1,$izi ] )*2 )) end
macro  harm_zi(A)  @expandargs(A);  esc(:(1/(1/$A[$ixi ,$iyi ,$iz  ] + 1/$A[$ixi ,$iyi ,$iz+1] )*2 )) end
macro harm_xya(A)  @expandargs(A);  esc(:(1/(1/$A[$ix  ,$iy  ,$iz  ] + 1/$A[$ix+1,$iy  ,$iz  ] +
                                               1/$A[$ix  ,$iy+1,$iz  ] + 1/$A[$ix+1,$iy+1,$iz  ] )*4 )) end
macro harm_xza(A)  @expandargs(A);  esc(:(1/(1/$A[$ix  ,$iy  ,$iz  ] + 1/$A[$ix+1,$iy  ,$iz  ] +
                                               1/$A[$ix  ,$iy  ,$iz+1] + 1/$A[$ix+1,$iy  ,$iz+1] )*4 )) end
macro harm_yza(A)  @expandargs(A);  esc(:(1/(1/$A[$ix  ,$iy  ,$iz  ] + 1/$A[$ix  ,$iy+1,$iz  ] +
                                               1/$A[$ix  ,$iy  ,$iz+1] + 1/$A[$ix  ,$iy+1,$iz+1] )*4 )) end
macro harm_xyi(A)  @expandargs(A);  esc(:(1/(1/$A[$ix  ,$iy  ,$izi ] + 1/$A[$ix+1,$iy  ,$izi ] +
                                               1/$A[$ix  ,$iy+1,$izi ] + 1/$A[$ix+1,$iy+1,$izi ] )*4 )) end
macro harm_xzi(A)  @expandargs(A);  esc(:(1/(1/$A[$ix  ,$iyi ,$iz  ] + 1/$A[$ix+1,$iyi ,$iz  ] +
                                               1/$A[$ix  ,$iyi ,$iz+1] + 1/$A[$ix+1,$iyi ,$iz+1] )*4 )) end
macro harm_yzi(A)  @expandargs(A);  esc(:(1/(1/$A[$ixi ,$iy  ,$iz  ] + 1/$A[$ixi ,$iy+1,$iz  ] +
                                               1/$A[$ixi ,$iy  ,$iz+1] + 1/$A[$ixi ,$iy+1,$iz+1] )*4 )) end
macro   maxloc(A)  @expandargs(A);  esc(:( max( max( max( max($A[$ixi-1,$iyi  ,$izi  ], $A[$ixi+1,$iyi  ,$izi  ])  , $A[$ixi  ,$iyi  ,$izi  ] ),
                                                          max($A[$ixi  ,$iyi-1,$izi  ], $A[$ixi  ,$iyi+1,$izi  ]) ),
                                                          max($A[$ixi  ,$iyi  ,$izi-1], $A[$ixi  ,$iyi  ,$izi+1]) ) )) end
macro   minloc(A)  @expandargs(A);  esc(:( min( min( min( min($A[$ixi-1,$iyi  ,$izi  ], $A[$ixi+1,$iyi  ,$izi  ])  , $A[$ixi  ,$iyi  ,$izi  ] ),
                                                          min($A[$ixi  ,$iyi-1,$izi  ], $A[$ixi  ,$iyi+1,$izi  ]) ),
                                                          min($A[$ixi  ,$iyi  ,$izi-1], $A[$ixi  ,$iyi  ,$izi+1]) ) )) end

@doc WITHIN_DOC
macro within(macroname::String, A)
    @expandargs(A)
    if     macroname == "@all"     esc(  :($ix<=size($A,1)   && $iy<=size($A,2)   && $iz<=size($A,3)  )  )
    elseif macroname == "@inn"     esc(  :($ix<=size($A,1)-2 && $iy<=size($A,2)-2 && $iz<=size($A,3)-2)  )
    elseif macroname == "@inn_x"   esc(  :($ix<=size($A,1)-2 && $iy<=size($A,2)   && $iz<=size($A,3)  )  )
    elseif macroname == "@inn_y"   esc(  :($ix<=size($A,1)   && $iy<=size($A,2)-2 && $iz<=size($A,3)  )  )
    elseif macroname == "@inn_z"   esc(  :($ix<=size($A,1)   && $iy<=size($A,2)   && $iz<=size($A,3)-2)  )
    elseif macroname == "@inn_xy"  esc(  :($ix<=size($A,1)-2 && $iy<=size($A,2)-2 && $iz<=size($A,3)  )  )
    elseif macroname == "@inn_xz"  esc(  :($ix<=size($A,1)-2 && $iy<=size($A,2)   && $iz<=size($A,3)-2)  )
    elseif macroname == "@inn_yz"  esc(  :($ix<=size($A,1)   && $iy<=size($A,2)-2 && $iz<=size($A,3)-2)  )
    else error("unkown macroname: $macroname. If you want to add your own assignement macros, overwrite the macro 'within(macroname::String, A)'; to still use the exising macro within as well call ParallelStencil.FiniteDifferences{1|2|3}D.@within(macroname, A) at the end.")
    end
end

end # Module FiniteDifferences3D
