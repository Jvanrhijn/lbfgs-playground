! Module with objective functions for optimization testing
module objectives
use olbfgs, only: olbfgs_iteration, update_hessian
implicit none
contains

  function rosenbrock(x, n) result(val)
    real(kind=9), dimension(1), intent(in) :: x
    integer), intent(in) :: n
    real(kind=8) :: val

    ! temp values
    real(kind=8) :: t1 = 0.d0
    real(kind=8) :: t2 = 0.d0

    ! counter
    integer :: j = 1

    val = 0.d0
    do j=1,n,2
      t1 = 1.d0 - x(j)
      t2 = 1.d1*(x(j + 1) - x(j)**2)
      val = val + t1**2.d0 + t2**2.d0
    end do
  end function

  subroutine rosenbrock_grad(x, n, g) 
    real(kind=8), dimension(:) :: g
    real(kind=8), dimension(:), intent(in) :: x
    integer, intent(in) :: n

    ! temp values
    real(kind=8) :: t1
    real(kind=8) :: t2

    ! counter
    integer :: j = 1

    do j=1,n,2
      t1 = 1.d0 - x(j)
      t2 = 1.d1*(x(j + 1) - x(j)**2)
      g(j + 1) = 2.d1 * t2
      g(j) = -2.d0 * (x(j) * g(j+1) + t1)
    end do

  end subroutine 

end module


program main
  use objectives
  use numeric_kinds
  implicit none
  
  integer, parameter :: dp = kind(0.d0)

  ! variable declarations
  integer :: i = 1 ! counter variable
  integer, parameter :: n = 100 ! number of dimensions
  integer, parameter :: m = 5 ! number of history points to keep

  ! solution array
  real(dp), dimension(:), allocatable :: x
  real(dp), dimension(:), allocatable :: y
  integer, parameter :: num_iters = 100

  ! function value, gradient vector 
  real(dp) :: f
  real(dp) :: g(n)
  real(dp) :: t1, t2

  ! mutable data for LBFGS
  real(dp) :: w(n*(2*m + 1) + 2*m) ! scratchpad
  real(dp) :: diag(n) ! hessian diagonal

  ! noise vector and value
  real(dp) :: rand(n)
  real(dp) :: noise

  ! step size annealing
  real(dp) :: step_naught = 0.00001d0
  real(dp) :: step
  real(dp) :: decay_const = 1.d0


  ! allocate solution array
  allocate(x(n))
  
  ! initial guess for x
  !x = (/(dble(i), i=1,n)/)
  do i=1,n,2
    x(i) = -1.2d0
    x(i+1) = 1.d0
  end do

  print '("Initial objective value: ", (d10.5))', rosenbrock(x, n)

  ! optimize the function
  optimize: do i=1, num_iters
    step = step_naught/(i - 1.d0 + decay_const)
    f = rosenbrock(x, n)
    call rosenbrock_grad(x, n, g)
    ! add noise to function value
    call random_number(noise)
    !f = f + 0.01d0*2.d0*(noise - 0.5d0)*abs(f)
    ! add noise to gradient vector
    call random_number(rand)
    !g = g + 0.01d0*2.d0*(rand - 0.5d0)*maxval(abs(g))
    ! perform LBFGS iteration
    call lbfgs_iteration(f, x, g, n, m, diag, w, step)
    print *, f
  end do optimize

  print '("Final objective function value: ", (d10.5))', rosenbrock(x, n)

  ! clean up
  deallocate(x)

end program main
