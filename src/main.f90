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
  integer(i4b) :: i = 1 ! counter variable
  integer(i4b), parameter :: n = 10 ! number of dimensions
  integer(i4b), parameter :: m = 5 ! number of history points to keep

  ! solution array
  real(dp), dimension(:), allocatable :: x
  integer(i4b), parameter :: num_iters = 99

  ! function value, gradient vector 
  real(dp) :: f
  real(dp) :: g(n)

  ! mutable data for LBFGS
  real(dp) :: w(n*(2*m + 1) + 2*m) ! scratchpad
  real(dp) :: diag(n) ! hessian diagonal

  ! allocate solution array
  allocate(x(n))
  
  ! initial guess for x
  x = (/(dble(i), i=1,n)/)

  print '("Initial objective value: ", (d10.5))', norm_square(x, n)

  ! optimize the function
  optimize: do i=1, num_iters
    f = norm_square(x, n)
    g = norm_square_grad(x, n)
    call lbfgs_iteration(f, x, g, n, m, diag, w)
    print *, f
  end do optimize

  print '("Final objective function value: ", (d10.5))', norm_square(x, n)

  ! clean up
  deallocate(x)

end program main
