! Module with objective functions for optimization testing
module objectives
use olbfgs, only: initialize_olbfgs, olbfgs_iteration, update_hessian
implicit none
contains

  function rosenbrock(x, n) result(val)
    real(kind=8), dimension(1), intent(in) :: x
    integer, intent(in) :: n
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
  implicit none
  
  ! variable declarations
  integer :: i = 1 ! counter variable
  integer, parameter :: n = 100 ! number of dimensions
  integer, parameter :: m = 10  ! number of history points to keep

  ! Noise level; increase for noisier objective function
  real(kind=8), parameter :: noise_level = 0.01

  ! solution array
  real(kind=8), dimension(:), allocatable :: x
  real(kind=8), dimension(:), allocatable :: y
  integer, parameter :: num_iters = 200

  ! function value, gradient vector 
  real(kind=8) :: f
  real(kind=8) :: g(n)

  real(kind=8), parameter :: step = 0.5d0

  ! noise vector and value
  real(kind=8) :: rand(n)
  real(kind=8) :: noise

  ! allocate solution array
  allocate(x(n))
  
  ! initial guess for x
  do i = 1, n, 2
    x(i) = -1.2d0
    x(i+1) = 1.d0
  end do

  !print '("Initial objective value: ", (d10.5))', rosenbrock(x, n)
  f = rosenbrock(x, n)
  call rosenbrock_grad(x, n, g)

  ! initialize o-LBFGS
  call initialize_olbfgs(x, g, n, m)

  ! optimize the function
  do i=1, num_iters
    ! perform LBFGS iteration
    call olbfgs_iteration(x, g, step, i)
    ! retrieve function value and gradient
    f = rosenbrock(x, n)
    call rosenbrock_grad(x, n, g)
    ! add noise to function value
    call random_number(noise)
    f = f + noise_level * 2.d0*(noise - 0.5d0)*abs(f)
    ! add noise to gradient vector
    call random_number(rand)
    g = g + noise_level * 2.d0*(rand - 0.5d0)*maxval(abs(g))
    ! update hessian approximation
    call update_hessian(x, g)
    ! output progress
    print *, f
  end do 

  !print '("Final objective function value: ", (d10.5))', rosenbrock(x, n)

  ! clean up
  deallocate(x)

end program
