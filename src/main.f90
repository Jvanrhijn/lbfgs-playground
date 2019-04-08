! Module with objective functions for optimization testing
module objectives
use numeric_kinds
use lbfgs_wrapper, only: lbfgs_iteration
implicit none
contains

  function norm_square(x, n)
    real(dp) :: x(:)
    integer(i4b) :: n
    real(dp) :: norm_square
    norm_square = sum(x**2)
  end function

  function norm_square_grad(x, n)
    real(dp), dimension(:) :: x
    integer(i4b) :: n
    ! output gradient vector
    real(dp), dimension(:), allocatable :: norm_square_grad 
    allocate(norm_square_grad(n))
    ! derivative of x^2:
    norm_square_grad = 2*x
  end function

  function rastrigin(x, n)
    real(dp) :: rastrigin
    real(dp) , dimension(1) :: x
    real(dp) , parameter :: pi = 3.14159265358979
    integer(i4b) :: n
    rastrigin = 10*n + sum(x**2 - 10.d0*cos(2*pi*x))
  end function

  function rastrigin_grad(x, n)
    integer(i4b) :: n
    real(dp), dimension(:) :: x
    real(dp), dimension(:), allocatable :: rastrigin_grad
    real(dp), parameter :: pi = 3.14159265358979
    allocate(rastrigin_grad(n))
    rastrigin_grad = 2.d0*x + 20.d0*pi*sin(2*pi*x)
  end function

end module


program main
  use objectives
  use numeric_kinds
  implicit none

  external LB2 ! absolutely horrifying

  ! variable declarations
  integer(i4b), parameter :: n = 10
  real(dp), dimension(:), allocatable :: x
  integer(i4b) :: i = 1
  integer(i4b), parameter :: num_iters = 99

  ! constant parameters for LBFGS algorithm
  real(dp) :: f
  real(dp) :: g(n)
  integer(i4b), parameter :: m = 5 ! number of history points to keep
  integer(i4b), dimension(1) :: iprint(2) ! printing data
  real(dp), parameter :: eps = 1.0d-5 ! don't use any termination criteria
  real(dp), parameter :: xtol = 1.0d-16 ! line search convergence criterion
  logical, parameter :: diagco = .false. ! don't provide diagonal matrices

  ! mutable data for LBFGS
  integer(i4b) :: iflag = 0 ! error flag
  real(dp) :: w(n*(2*m + 1) + 2*m)
  real(dp) :: diag(n)

  ! set print specifications
  iprint(1) = -1 ! don't print anything
  iprint(2) = 0 ! arbitrary since we don't print anything

  ! allocate solution array
  allocate(x(n))
  
  ! initial guess for x
  x = (/(dble(i), i=1,n)/)

  print '("Initial objective value: ", (d10.5))', norm_square(x, n)

  ! optimize the function
  optimize: do i=1, num_iters
    f = norm_square(x, n)
    g = norm_square_grad(x, n)
    !call lbfgs(n, m, x, f, g, diagco, diag, iprint, eps, xtol, w, iflag)
    call lbfgs_iteration(f, x, g, n, m, diag, w)
    print *, f
  end do optimize

  print '("Final objective function value: ", (d10.5))', norm_square(x, n)

  ! clean up
  deallocate(x)

end program main
