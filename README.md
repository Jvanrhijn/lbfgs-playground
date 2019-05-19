# LBFGS playground

This repository contains a Fortran 90 implementation of the
"online" L-BFGS (o-LBFGS) algorithm outlined in 
[Schraudolf (2007)](http://proceedings.mlr.press/v2/schraudolph07a/schraudolph07a.pdf).

## Usage

Assume in the following that `n` denotes the number of variables `x` to
optimize, `m` denotes the number of curvature pairs to store, `f` denotes the
objective function, and `g` its gradient. Then, a minimal usage example is
as follows.

~~~fortran
program main
    use olbfgs, only: initialize_olbfgs, olbfgs_iteration, update_hessian
    implicit none

    ! algorithm parameters
    integer :: n = 100
    integer :: m = 5
    double precision :: step_size = 0.5

    double precision :: value

    double precision, dimension(:) :: x
    double precision, dimension(:) :: grad

    ! initialize parameter vector x
    x = ...

    ! First intiialize the algorithm (i.e. allocate storage)
    call initialize_olbfgs(n, m)

    do i = 1, 100, 1
        ! output function value
        print *, f(x)
        ! compute function value, gradient
        value = f(x)
        grad = g(x)
        ! update the Hessian approximation
        call update_hessian(x, g)
        ! perform optimization iteration
        call olbfgs_iteration(x, g, step_size, i)
    end do

end program
~~~

The call to `initialize_olbfgs` before calling either
`update_hessian` or `olbfgs_iteration` is required for allocating
memory used internally in the module.