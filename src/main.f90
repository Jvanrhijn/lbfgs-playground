! Module with objective functions for optimization testing
module objectives
use numeric_kinds
use lbfgs_wrapper, only: lbfgs_iteration
implicit none
contains

  function rosenbrock(x, n) result(val)
    real(dp), dimension(1), intent(in) :: x
    integer(i4b), intent(in) :: n
    real(dp) :: val

    ! temp values
    real(dp) :: t1 = 0.d0
    real(dp) :: t2 = 0.d0

    ! counter
    integer(i4b) :: j = 1

    val = 0.d0
    do j=1,n,2
      t1 = 1.d0 - x(j)
      t2 = 1.d1*(x(j + 1) - x(j)**2.d0)
      val = val + t1**2.d0 + t2**2.d0
    end do
  end function

  function rosenbrock_grad(x, n) result(g)
    real(dp), dimension(:), allocatable :: g
    real(dp), dimension(:) :: x
    integer(i4b) :: n

    ! temp values
    real(dp) :: t1 = 0.d0
    real(dp) :: t2 = 0.d0

    ! counter
    integer(i4b) :: j = 1

    allocate(g(n))

    do j=1,n,2
      t1 = 1.d0 - x(j)
      t2 = 1.d0*(x(j + 1) - x(j)**2)
      g(j + 1) = 2.d1 * t2
      g(j) = -2.d0 * (x(j) * g(j+1) + t1)
    end do

  end function

end module


program main
  use objectives
  use numeric_kinds
  implicit none

  external LB2 ! absolutely horrifying

  ! variable declarations
  integer(i4b) :: i = 1 ! counter variable
  integer(i4b), parameter :: n = 100 ! number of dimensions
  integer(i4b), parameter :: m = 5 ! number of history points to keep

  ! solution array
  real(dp), dimension(:), allocatable :: x
  real(dp), dimension(:), allocatable :: y
  integer(i4b), parameter :: num_iters = 30

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

  print '("Initial objective value: ", (d10.5))', rosenbrock(x, n)

  ! optimize the function
  optimize: do i=1, num_iters
    f = rosenbrock(x, n)
    g = rosenbrock_grad(x, n)
    call lbfgs_iteration(f, x, g, n, m, diag, w)
    print *, f
  end do optimize

  print '("Final objective function value: ", (d10.5))', rosenbrock(x, n)

  ! clean up
  deallocate(x)

end program main
