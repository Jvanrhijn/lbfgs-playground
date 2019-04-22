module olbfgs
implicit none

    private 
    
        integer, parameter :: dp = kind(0.d0)
        real(dp), allocatable :: s(:, :)
        real(dp), allocatable :: y(:, :)
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
        if (.not. (allocated(s) &
             .and. allocated(y) &
             .and. allocated(gradient_prev))) then
            allocate(s(history_size, num_pars))
            allocate(y(history_size, num_pars))
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
        integer :: m 
        m = size(s(:, 1))
        s(curvature_index, :) = parameters - parms_prev
        y(curvature_index, :) = gradient - gradient_prev
        curvature_index = curvature_index + 1
        curvature_index = mod(curvature_index - 1 , m) + 1
    end subroutine

    function initial_direction(gradient, iteration)
        real(dp), allocatable :: initial_direction(:) 
        real(dp), intent(in) :: gradient(:)
        integer, intent(in) :: iteration

        ! local variables
        integer :: n, m, i, head
        real(dp) :: alpha, beta, tot
        real(dp), allocatable :: alphas(:)
        n = size(gradient) ! number of parameters
        m = size(s(:, 1)) ! history size

        ! allocate return value
        allocate(initial_direction(n))

        ! allocate local data
        allocate(alphas(min(m, iteration)))

        ! perform hessian approximation using history
        initial_direction = -gradient

        ! first of two-loop recursion
        ! TODO: figure out way to use curvature_meow as queue
        if (iteration > 1) then
            do i=1, min(m, iteration-1)
                head = curvature_index - i
                if (head < 1) then
                    head = min(m, iteration-1) - i + 1
                end if
                alpha = dot_product(s(head, :), initial_direction) &
                    / dot_product(s(head, :), y(head, :))
                initial_direction = initial_direction - alpha * y(head, :)
                alphas(i) = alpha
            end do
        end if

        ! scale search direction
        if (iteration == 1) then
            initial_direction = initial_direction * eps
        else
            tot = 0.d0
            do i=1, min(m, iteration-1)
                tot = tot + dot_product(s(i, :), y(i, :)) &
                    / dot_product(y(i, :), y(i, :))
            end do
            initial_direction = initial_direction * tot/min(m, iteration)
        end if

        ! second of two-loop recursion
        if (iteration > 1) then
            do i=1, min(m, iteration-1)
                alpha = alphas(size(alphas) + 1 - i)
                head = curvature_index - min(m, iteration-1) + i
                if (head < 1) then
                    head = curvature_index + i
                end if
                print *, head
                beta = dot_product(y(head, :), initial_direction) &
                    / dot_product(y(head, :), s(head, :))
                initial_direction = initial_direction + (alpha - beta) * s(head, :)
            end do
        end if

    end function 

end module