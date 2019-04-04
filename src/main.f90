! Module with objective functions for optimization testing
module objectives
implicit none
contains

  function norm_square(x, n)
    double precision :: x(:)
    integer :: n
    double precision :: norm_square
    norm_square = sum(x**2)
  end function

  function norm_square_grad(x, n)
    double precision, dimension(:) :: x
    integer :: n

    ! output gradient vector
    double precision, dimension(:), allocatable :: norm_square_grad 
    allocate(norm_square_grad(n))

    ! derivative of x^2:
    norm_square_grad = 2*x
  end function

  function rastrigin(x, n)
    double precision :: rastrigin
    double precision, dimension(1) :: x
    double precision, parameter :: pi = 3.14159265358979
    integer :: n
    rastrigin = 10*n + sum(x**2 - 10.d0*cos(2*pi*x))
  end function

  function rastrigin_grad(x, n)
    integer :: n
    double precision, dimension(:) :: x
    double precision, dimension(:), allocatable :: rastrigin_grad
    double precision, parameter :: pi = 3.14159265358979
    allocate(rastrigin_grad(n))
    rastrigin_grad = 2.d0*x + 20.d0*pi*sin(2*pi*x)
  end function

end module


program main
  use objectives
  implicit none

  external LB2 ! absolutely horrifying

  ! variable declarations
  integer, parameter :: n = 10
  double precision, dimension(:), allocatable :: x
  integer :: i = 1
  integer, parameter :: num_iters = 100

  ! constant parameters for LBFGS algorithm
  double precision :: f
  double precision :: g(n)
  integer, parameter :: m = 5 ! number of history points to keep
  integer, dimension(1) :: iprint(2) ! printing data
  double precision, parameter :: eps = 1.0d-5 ! don't use any termination criteria
  double precision, parameter :: xtol = 1.0d-16 ! line search convergence criterion
  logical, parameter :: diagco = .false. ! don't provide diagonal matrices

  ! mutable data for LBFGS
  integer :: iflag = 0 ! error flag
  double precision :: w(n*(2*m + 1) + 2*m)
  double precision :: diag(n)

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
    call lbfgs(n, m, x, f, g, diagco, diag, iprint, eps, xtol, w, iflag)
    if (iflag == -1) then
      exit optimize
    end if
  end do optimize

  print '("Final objective function value: ", (d10.5))', norm_square(x, n)

  ! clean up
  deallocate(x)

end program main
