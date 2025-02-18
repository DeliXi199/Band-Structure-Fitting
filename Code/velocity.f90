module para
    implicit none
    real(8), allocatable, dimension(:, :, :) :: star_R  ! 三维数组
    real(8), allocatable, dimension(:,:) :: coes  ! 系数数组
    integer, dimension(4) :: NMS  ! 用于存储输入的数组维度信息
    integer, allocatable, dimension(:) :: star_R_len
    real(8) :: pi = 3.14159265358979323846d0
end module para


program Velocity_program
    use para
    implicit none
    
    real(8),DIMENSION(3)::kkk
    real(8),DIMENSION(3)::velocity_vector
    real(8) :: my_energy
    INTEGER :: iband
    real :: start_time, end_time, elapsed_time
    INTEGER :: i,j,k,N
    real, ALLOCATABLE,dimension(:,:) :: arr_velocity
    real, ALLOCATABLE,dimension(:) :: arr_E
    real, ALLOCATABLE,dimension(:) :: arr_E1
    real, ALLOCATABLE,dimension(:) :: arr_E2
    real :: step, start_value

    ! 删除不需要的生成文件
    character(len=100) :: command
    integer :: ierr

    iband = 1
    N=50

    ALLOCATE(arr_velocity(N,3))
    ALLOCATE(arr_E(N))
    ALLOCATE(arr_E1(N))
    ALLOCATE(arr_E2(N))

    call read_txt_file()  ! 读取数据文件

    ! 打印 star_R 的尺寸信息
    print *, "star_R size:", size(star_R, 1), size(star_R, 2), size(star_R, 3)

    ! 计算 kkk 并调用 velocity 函数
    kkk = (/ 0.0, 0.0000000e00, 0.0000000e00 /) * 2 * pi  ! 定义 kkk 向量
    print *, "kkk:", kkk

    call my_velocity(iband,kkk, velocity_vector)  ! 调用速度计算函数
    ! 打印结果
    print *, "v:", velocity_vector


    call my_E(iband,kkk, my_energy)
    print *, "E:", my_energy

    ! 获取程序开始的时间
    call cpu_time(start_time)

    ! 模拟一些计算任务

    do i = 1, N
        kkk=(/ 1.0, 0.0000000e00, 0.0000000e00 /)  * pi *i/N
        call my_velocity(iband,kkk, velocity_vector)  ! 调用速度计算函数
        print *, velocity_vector
        arr_velocity(i,:) = velocity_vector  ! 填充数组
        call my_E(iband,kkk+(/ 0.005, 0.0000000e00, 0.0000000e00 /) * 2 * pi, my_energy)
        arr_E1(i) = my_energy  ! 填充数组
        call my_E(iband,kkk+(/ -0.005, 0.0000000e00, 0.0000000e00 /) * 2 * pi, my_energy)
        arr_E2(i) = my_energy  ! 填充数组
    end do
    ! 获取程序结束的时间
    call cpu_time(end_time)

    ! 计算和输出经过的时间
    elapsed_time = end_time - start_time
    print *, "Elapsed CPU time: ", elapsed_time, " seconds"

      ! 循环填充三维数组

    ! 写入TXT文件
    open(unit=10, file='./../Data/mesh10.txt', status='replace')
    do i = 1, N
        write(10, *) arr_velocity(i, :)
    end do
    close(10)

    arr_E = (arr_E1+arr_E2)/2
    open(unit=10, file='./../Data/energy_k.txt', status='replace')
    do i = 1, N
        write(10, *) arr_E(i)
    end do
    close(10)

    open(unit=10, file='./../Data/v_minus.txt', status='replace')
    do i = 1, N
        write(10, *) (arr_E1(i)-arr_E2(i))/(0.01*2*pi)
    end do
    close(10)

    ! 运行结束后的文件删除命令
    ! 删除 .exe 文件
    write(command, '("del /f ", A)') 'velocity.exe'
    call execute_command_line(command, exitstat=ierr)

    ! 删除 .mod 文件
    write(command, '("del /f ", A)') 'para.mod'
    call execute_command_line(command, exitstat=ierr)


CONTAINS


    subroutine read_txt_file()
        use para
        implicit none

        integer :: i, j, k, nx, ny, nz, Nband_Fermi_Level


        ! 读取 NMS.txt 文件
        open(unit=1, file='./../Data/NMS.txt', status='old', action='read')
        read(1, *) NMS  ! 从文件中读取 NMS 数组
        close(1)

        ! 初始化数组维度
        nx = NMS(2)
        ny = NMS(3)
        nz = 3
        Nband_Fermi_Level = NMS(4)

        ! 分配并读取 star_R 数组
        allocate(star_R(nx, ny, nz))
        open(unit=2, file='./../Data/star_R.txt', status='old', action='read')
        do i = 1, nx
            do j = 1, ny
                read(2, *) (star_R(i, j, k), k = 1, nz)  ! 从文件中逐行读取
            end do
        end do
        close(2)

        allocate(star_R_len(nx))
        open(unit=3, file='./../Data/star_R_len.txt', status='old', action='read')
        do i = 1, nx
            read(3, *) star_R_len(i)  ! 从文件中逐行读取
        end do
        close(3)

        nx = NMS(2)
        ! 分配并读取 coes 数组
        allocate(coes(Nband_Fermi_Level,nx))
        open(unit=4, file='./../Data/coes.txt', status='old', action='read')
        do i = 1, Nband_Fermi_Level
            read(4, *) (coes(i, j), j = 1, nx)  ! 从文件中逐行读取
        end do
        close(4)   

        return
    end subroutine read_txt_file

    subroutine get_star_R_m(m, star_R_m)

        use para
        implicit none
        integer :: m
        real(8), dimension(:,:), allocatable :: star_R_m
        INTEGER :: len_star_R_m


        ! 获取 star_R_m 的长度
        len_star_R_m = star_R_len(m)

        DEALLOCATE(star_R_m)  ! 释放 star_R_m 的空间

        ! 分配 star_R_m 的空间
        allocate(star_R_m(len_star_R_m, 3))

        ! 提取 star_R_m 子数组
        star_R_m = star_R(m, 1:len_star_R_m, :)
        ! print *, "size of star_R_m: ", size(star_R_m, 1), size(star_R_m, 2)
    end subroutine get_star_R_m

    ! 计算速度的函数
    subroutine my_velocity(iband,kkk, velocity_vector)
        use para

        implicit none
        INTEGER :: iband
        real(8), allocatable, dimension(:,:) :: star_R_m  ! 临时二维矩阵
        real(8), dimension(3), intent(in) :: kkk  ! 波矢向量
        real(8), dimension(3) :: v  ! 输出速度向量
        real(8), dimension(3) :: v_m  ! 临时速度向量
        integer :: m, l
        real(8), dimension(3), intent(out) :: velocity_vector       ! 速度向量

        real(8), DIMENSION(:), ALLOCATABLE :: coe

        ALLOCATE(coe(NMS(2)))
        coe = coes(iband,:)
        allocate(star_R_m(NMS(3), 3))  ! 分配二维数组


        v = (/ 0.0d0, 0.0d0, 0.0d0 /)  ! 初始化速度向量

        do m = 2, NMS(2)
            
            ! call get_star_R_m(m, star_R_m)  ! 提取 star_R_m 子数组
            star_R_m = star_R(m, :, :)  ! 提取 star_R_m 子数组
            ! print *, "size of star_R_m: ", size(star_R_m, 1), size(star_R_m, 2)

            v_m = (/ 0.0d0, 0.0d0, 0.0d0 /)  ! 初始化临时速度向量
            do l = 1, star_R_len(m)
                v_m = v_m - sin(dot_product(star_R_m(l, :), kkk)) * star_R_m(l, :)  ! 累积计算速度
            end do
            v = v + coe(m) * v_m / star_R_len(m) *2 ! 更新速度向量
        end do

        velocity_vector = v  ! 输出速度向量
    end subroutine my_velocity

    subroutine my_E(iband,kpointsk,Eki)
        use para

        implicit none
        integer :: iband
        real(8), allocatable, dimension(:,:) :: star_R_m  ! 临时二维矩阵
        real(8), dimension(3), intent(in) :: kpointsk  ! 波矢向量
        real(8):: E  ! 输出速度向量
        real(8):: E_m  ! 临时速度向量
        integer :: m, l,i
        real(8), intent(out) :: Eki       ! 速度向量
        real(8), DIMENSION(:), ALLOCATABLE :: coe

        ALLOCATE(coe(NMS(2)))
        coe = coes(iband,:)
        allocate(star_R_m(NMS(3), 3))  ! 分配二维数组


        E = 0.0d0

        do m = 2, NMS(2)
            ! call get_star_R_m(m, star_R_m)  ! 提取 star_R_m 子数组
            ! print *, "size of star_R_m: ", size(star_R_m, 1), size(star_R_m, 2)
            ! if (m == 4) then
            !     print *, "size of star_R_m: ", size(star_R_m, 1), size(star_R_m, 2)
            !     do i= 1, size(star_R_m, 1)
            !         print *, "star_R_m: ", star_R_m(i,:)
            !     end do
            ! end if
            star_R_m = star_R(m, :, :)  ! 提取 star_R_m 子数组

            E_m = 0.0d0 ! 初始化临时速度向量
            do l = 1, star_R_len(m)
                E_m = E_m + cos(dot_product(star_R_m(l, :), kpointsk))   ! 累积计算速度
            end do
            E = E + coe(m) * E_m / star_R_len(m) ! 更新速度向量
            ! print *, "E: ", E
        end do

        do m = 1,1
            ! call get_star_R_m(m, star_R_m)  ! 提取 star_R_m 子数组
            star_R_m = star_R(m, :, :)  ! 提取 star_R_m 子数组

            E_m = 0.0d0  ! 初始化临时速度向量
            do l = 1, star_R_len(m)
                E_m = E_m + cos(dot_product(star_R_m(l, :), kpointsk))   ! 累积计算速度
            end do
            ! print *, "star_R_m: ", star_R_m
            E = E + coe(m) * E_m / star_R_len(m) ! 更新速度向量
            deallocate(star_R_m)  ! 释放临时数组
        end do

        Eki = E  ! 输出速度向量
    end subroutine my_E




end program Velocity_program
