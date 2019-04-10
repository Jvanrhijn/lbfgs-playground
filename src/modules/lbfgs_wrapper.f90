!> This module is a Fortran 90 wrapper around the LBFGS
!> algorithm as implemented by Nodecal.

module lbfgs_wrapper

use numeric_kinds, only: dp, i4b
implicit none

    ! default scalar values for LBFGS
    real(dp), parameter :: xtol = 1.0d-16 ! line search convergence criterion
    real(dp), parameter :: eps = 1.0d-5 ! convergence criterion for LBFGS
    logical, parameter :: diagco = .false. ! don't provide diagonal matrix
    integer(i4b), dimension(2) :: iprint(2) ! printing configuration

contains

    subroutine lbfgs_iteration(function_value, parameters, gradient, num_pars, history_size, diag, workspace)
        ! input parameters
        real(dp), intent(in) :: function_value
        real(dp), intent(in) :: gradient(num_pars)
        integer(i4b), intent(in) :: num_pars
        integer(i4b), intent(in) :: history_size

        ! mutable parameters
        real(dp), intent(inout) :: parameters(num_pars)
        real(dp), intent(inout) :: workspace(num_pars*(2*history_size + 1) + 2*history_size)
        real(dp), intent(inout) :: diag(num_pars)
        integer(i4b) :: error_flag = 0

        ! printing configuration
        iprint(1) = -1 ! don't print LBFGS info
        iprint(2) = 0 ! arbitrary since we're not printing anything

        call lbfgs(num_pars, history_size, parameters, function_value, gradient, &
                   diagco, diag, iprint, eps, xtol, workspace, error_flag)

    end subroutine

end module