module olbfgs
implicit none

    private 
    
        integer, parameter :: dp = kind(0.d0)
        real(dp), allocatable :: curvature_s(:, :)
        real(dp), allocatable :: curvature_y(:, :)
        real(dp), allocatable :: gradient_prev(:)
        real(dp), allocatable :: parms_prev(:)
        real(dp), parameter :: eps = 1.0e-10_dp
        integer :: curvature_index = 1

    public olbfgs_iteration, update_hessian

contains

    subroutine olbfgs_iteration(num_pars, history_size, parameters, gradient, step_size, iteration)
        integer, intent(in) :: num_pars
        integer, intent(in) :: history_size
        real(dp), intent(inout) :: parameters(:)
        real(dp), intent(inout) :: gradient(:)
        real(dp), intent(in) :: step_size
        integer, intent(in) :: iteration

        ! local data
        real(dp), allocatable :: p(:)
        integer :: i

        ! initial setup upon first call
        if (.not. (allocated(curvature_s) &
             .and. allocated(curvature_y) &
             .and. allocated(gradient_prev))) then
            allocate(curvature_s(history_size, num_pars))
            allocate(curvature_y(history_size, num_pars))
            allocate(gradient_prev(num_pars))
            allocate(parms_prev(num_pars))
            gradient_prev = (/(0, i=1, num_pars)/)
            parms_prev = (/(0, i=1, num_pars)/)
        end if

        ! allocate local data
        allocate(p(num_pars))

        ! compute initial search direction
        p = initial_direction(gradient, iteration)

        ! save previous gradient and parameters value
        gradient_prev = gradient
        parms_prev = parameters

        ! update parmaters
        parameters = parameters + step_size * p

    end subroutine

    subroutine update_hessian(parameters, gradient)
        real(dp), dimension(:), intent(in) :: parameters
        real(dp), dimension(:), intent(in) :: gradient
        curvature_s(curvature_index, :) = parameters - parms_prev
        curvature_y(curvature_index, :) = gradient - gradient_prev
    end subroutine

    pure function initial_direction(gradient, iteration)
        real(dp), allocatable :: initial_direction(:) 
        real(dp), intent(in) :: gradient(:)
        integer, intent(in) :: iteration

        ! local variables
        integer :: n, m, i
        real(dp) :: alpha, beta, tot
        real(dp), allocatable :: alphas(:)
        n = size(gradient) ! number of parameters
        m = size(curvature_s(:, 1)) ! history size

        ! allocate return value
        allocate(initial_direction(n))

        ! allocate local data
        allocate(alphas(min(m, iteration)))

        ! perform hessian approximation using history
        initial_direction = -gradient

        ! first of two-loop recursion
        ! TODO figure out way to use curvature_meow as queue
        do i=min(m, iteration), 1, -1
            alpha = dot_product(curvature_s(i, :), initial_direction) &
                / dot_product(curvature_s(i, :), curvature_y(i, :))
            initial_direction = initial_direction - alpha * curvature_y(i, :)
            alphas(i) = alpha
        end do

        ! scale search direction
        if (iteration == 0) then
            initial_direction = initial_direction * eps
        else
            tot = 0.d0
            do i=1, min(m, iteration)
                tot = tot + dot_product(curvature_s(i, :), curvature_y(i, :)) &
                    / dot_product(curvature_y(i, :), curvature_y(i, :))
            end do
            initial_direction = initial_direction * tot/min(m, iteration)
        end if

        ! second of two-loop recursion
        do i=1, min(m, iteration)
            alpha = alphas(size(alphas)+ 1 - i)
            beta = dot_product(curvature_y(i, :), initial_direction) &
                / dot_product(curvature_y(i, :), curvature_s(i, :))
            initial_direction = initial_direction + (alpha - beta) * curvature_s(i, :)
        end do

    end function 

end module