! Module with objective functions for optimization testing
module objectives
implicit none
contains

  function norm_square(x)
    double precision :: x(:)
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

end module


program main
  use objectives
  implicit none

  ! variable declarations
  integer, parameter :: n = 10
  double precision, dimension(:), allocatable :: x
  integer :: i = 1

  ! allocate solution array
  allocate(x(n))
  
  ! initial guess for x
  x = (/(dble(2*i), i=-n,n)/)

  print '("Initial objective value: ", (d10.5))', norm_square(x)

  ! optimization code here


  ! clean up
  deallocate(x)

end program main
