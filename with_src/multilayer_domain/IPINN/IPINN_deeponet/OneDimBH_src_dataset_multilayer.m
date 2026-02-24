function [u1, x, t, u_meas_02, u_meas_05, u_meas_08] = OneDimBH_src_dataset_multilayer(x_interface)
    m = 0;
    x = linspace(0,1,101);
    t = linspace(0,1,101);

    % ========================
    % Solve PDE
    % ========================
    sol = pdepe(m, @OneDimBHpde, @OneDimBHic, @OneDimBHbc, x, t);
    u1 = sol(:,:,1);

    % ========================
    % Base directories
    % ========================
    base_dir = "/MATLAB Drive/Tesi Magistrale";

    gt_dir     = fullfile(base_dir, "gt_bioheat1D_src_multilayer");
    fig_dir    = fullfile(base_dir, "gt_figures");
    u_meas_dir = fullfile(base_dir, "u_meas");

    if ~exist(gt_dir,"dir"),     mkdir(gt_dir);     end
    if ~exist(fig_dir,"dir"),    mkdir(fig_dir);    end
    if ~exist(u_meas_dir,"dir"), mkdir(u_meas_dir); end

    % ========================
    % Save ground truth CSV
    % ========================
    gt_filename = sprintf( ...
        'gt_bioheat1D_src_multilayer_xint_%.3f.csv', x_interface);

    fileID = fopen(fullfile(gt_dir, gt_filename),'w');
    for i = 1:length(t)
        for j = 1:length(x)
            fprintf(fileID,'%6.2f %6.2f %12.8f\n', x(j), t(i), u1(i,j));
        end
    end
    fclose(fileID);

    % ========================
    % Save figure
    % ========================
    fig = figure('Visible','off');

    surf(x, t, u1, 'EdgeColor', 'none'); 
    xlabel('x'); 
    ylabel('t'); 
    zlabel('u1(x,t)');
    
    title(sprintf('1D Bioheat Solution – x_{int} = %.3f', x_interface));
    colorbar;
    view(45,30);  
    
    fig_filename = sprintf('bioheat1D_3D_xint_%.3f.png', x_interface);
    saveas(fig, fullfile(fig_dir, fig_filename));
    close(fig);

    % ========================
    % Measurement points
    % ========================
    [~, ix02] = min(abs(x - 0.2));
    [~, ix05] = min(abs(x - 0.5));
    [~, ix08] = min(abs(x - 0.8));

    u_meas_02 = u1(:,ix02);
    u_meas_05 = u1(:,ix05);
    u_meas_08 = u1(:,ix08);

    % ========================
    % Estimate optimal T0
    % ========================
    T0_optimal_02 = extract_optimal_T0(t, u_meas_02);
    T0_optimal_05 = extract_optimal_T0(t, u_meas_05);
    T0_optimal_08 = extract_optimal_T0(t, u_meas_08);

    fprintf('Estimated optimal T0s from u_meas: [%.4f %.4f %.4f]\n', ...
        T0_optimal_02, T0_optimal_05, T0_optimal_08);

    % ========================
    % Save u_meas CSV
    % ========================
    u_meas_filename = sprintf( ...
    'u_meas_src_multilayer_xint_%.3f.csv', x_interface);

    fileID_u = fopen(fullfile(u_meas_dir, u_meas_filename),'w');
    
    fprintf(fileID_u,'t,u_x_0_2,u_x_0_5,u_x_0_8\n');

    for i = 1:length(t)
        fprintf(fileID_u,'%.8f,%.8f,%.8f,%.8f\n', ...
            t(i), u_meas_02(i), u_meas_05(i), u_meas_08(i));
    end

    % ========================
    % Nested functions
    % ========================
    function [c, f, s] = OneDimBHpde(x, t, u, dudx)
        wb = 0.0005;

        beta = 1;
        P = 25;
        t_span = 1800.035;

        c_vec  = [2348,3421];
        ro_vec = [911,1090];
        k_vec  = [0.21,0.49];

        Tmin = 21.5; x0 = 0.004; PD = 0.0136; L0 = 0.07;
        b4 = 0.829; y2_0 = 30.2;

        deltaT = (y2_0 - Tmin)/b4;
        v = log(2/(PD - 1e-2 * x0));

        if x <= x_interface
            i = 1;
        else
            i = 2;
        end

        a1 = (L0^2 * ro_vec(i) * c_vec(i)) / (k_vec(i) * t_span);
        a2 = L0^2 * ro_vec(i) * c_vec(i) / k_vec(i);
        a3 = (ro_vec(i)*L0^2/(k_vec(i)*deltaT)) * beta * exp(v*x0);
        a4 = v * L0;

        c = a1;
        f = dudx;
        s = -wb * a2 * u + P*a3 * exp(-a4*x);
    end

    function u0 = OneDimBHic(x)
        u0 = 0;
    end

    function [pl, ql, pr, qr] = OneDimBHbc(xl, ul, xr, ur, t)
        [~, y1, ~, ~] = ic_bc_src(xr, t);
        [~, ~, ~, y3] = ic_bc_src(xl, t);

        a5 = 175;
        pl = -a5 * (-y3 + ul);
        ql = 1;

        pr = ur - y1;
        qr = 0;
    end
end

function [T0] = extract_optimal_T0(t,u)
    myfittype = fittype('(1-exp(-t/T0))', ...
        'independent','t','dependent','u','coefficients',{'T0'});
    fitresult = fit(t(:), u(:), myfittype, 'StartPoint', 0.1);
    T0 = fitresult.T0;
end