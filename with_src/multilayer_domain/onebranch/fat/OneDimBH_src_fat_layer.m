function [sol] = OneDimBH_src_fat_layer()

  m = 0;
  x = linspace(0,1,101);
  t = linspace(0,1,101);

  % Run PDE solver with nested functions
  sol = pdepe(m, @OneDimBHpde, @OneDimBHic, @OneDimBHbc, x, t);

  u1 = sol(:,:,1); % solution of system

  % Write solution to file
  fileID = fopen("C:\Users\Mattia\Dropbox\Mattia\Tesi Magistrale/gt_bioheat1D_src_fat_layer.csv", 'w');
  for i = 1:101
      for j = 1:101
          fprintf(fileID,'%6.2f %6.2f %12.8f\n', x(j), t(i), u1(i,j));
      end
  end
  fclose(fileID);

% Extract solution at a specific x points within the domain
x_meas_02 = 0.2; x_meas_05 = 0.5; x_meas_08 = 0.8;
u_meas_02 = u1(:,(x == x_meas_02));
u_meas_05 = u1(:,(x == x_meas_05));
u_meas_08 = u1(:,(x == x_meas_08));


% Plot the solution u1 vs x and t
fig = figure();
surf(x, t, u1, 'EdgeColor', 'none');
xlabel('x');
ylabel('t');
zlabel('u1(x,t)');
beta = 1;
P =25;
title(sprintf('Solution of 1D Bioheat Equation (beta = %.2f), (Power = %.2f)', beta, P));
colorbar;
saveas(fig, 'bioheat_1D_src_fat_layer.png');

% Estimate optimal T0 from u_meas data
T0_optimal_02 = extract_optimal_T0(t, u_meas_02);
T0_optimal_05 = extract_optimal_T0(t, u_meas_05);
T0_optimal_08 = extract_optimal_T0(t, u_meas_08);
fprintf('Estimated optimal T0s from u_meas: [%.4f %.4f %.4f]\n', T0_optimal_02, T0_optimal_05, T0_optimal_08);

% Save u_meas vs t
fileID_u_02 = fopen("C:\Users\Mattia\Dropbox\Mattia\Tesi Magistrale/u_meas_src_fat_layer_02.csv", 'w');
fileID_u_05 = fopen("C:\Users\Mattia\Dropbox\Mattia\Tesi Magistrale/u_meas_src_fat_layer_05.csv", 'w');
fileID_u_08 = fopen("C:\Users\Mattia\Dropbox\Mattia\Tesi Magistrale/u_meas_src_fat_layer_08.csv", 'w');
  for i = 1:101
    fprintf(fileID_u_02,'%6.2f %6.2f %12.8f\n', t(i), u1(i,(x == x_meas_02))); 
    fprintf(fileID_u_05,'%6.2f %6.2f %12.8f\n', t(i), u1(i,(x == x_meas_05)));
    fprintf(fileID_u_08,'%6.2f %6.2f %12.8f\n', t(i), u1(i,(x == x_meas_08)));
  end
  fclose(fileID_u_02);
  fclose(fileID_u_05);
  fclose(fileID_u_08);

% Plot u_meas and fitted curves
fig2 = figure('Position', [100, 100, 1500, 400]);
subplot(1,3,1);
plot(t, u_meas_02, 'b-', 'LineWidth', 2);
hold on;
u_fit_02 = 1 - exp(-t / T0_optimal_02);
plot(t, u_fit_02, 'r--', 'LineWidth', 2);
legend('u_{meas}(t)', 'u_{fit}(t)', 'Location', 'Best');
xlabel('t');
ylabel(sprintf('u1(x=%.2f,t)', x_meas_02));
title(sprintf('Solution of 1D Bioheat Equation at x=%.2f, with T0=%.2f', x_meas_02, round(T0_optimal_02, 2)));
grid on;

subplot(1,3,2);
plot(t, u_meas_05, 'b-', 'LineWidth', 2);
hold on;
u_fit_05 = 1 - exp(-t / T0_optimal_05);
plot(t, u_fit_05, 'r--', 'LineWidth', 2);
legend('u_{meas}(t)', 'u_{fit}(t)', 'Location', 'Best');
xlabel('t');
ylabel(sprintf('u1(x=%.2f,t)', x_meas_05));
title(sprintf('Solution of 1D Bioheat Equation at x=%.2f, with T0=%.2f', x_meas_05, round(T0_optimal_05, 2)));
grid on;

subplot(1,3,3);
plot(t, u_meas_08, 'b-', 'LineWidth', 2);
hold on;
u_fit_08 = 1 - exp(-t / T0_optimal_08);
plot(t, u_fit_08, 'r--', 'LineWidth', 2);
legend('u_{meas}(t)', 'u_{fit}(t)', 'Location', 'Best');
xlabel('t');
ylabel(sprintf('u1(x=%.2f,t)', x_meas_08));
title(sprintf('Solution of 1D Bioheat Equation at x=%.2f, with T0=%.2f', x_meas_08, round(T0_optimal_08, 2)));
grid on;

saveas(fig2, 'u_meas_src_multilayer.png');


   % --------------------------------------------------------------------------
    function [c, f, s] = OneDimBHpde(x, t, u, dudx)
            wb = 0.0005;  % Uses config from outer scope

            %Parameters for multi-layer model
            beta = 1;
            P = 25;
            h = 525.0; t_span = 1800.035;
            c = 2348; ro = 911; k = 0.21; % fat 
            Tmin = 21.5; x0 = 0.004; PD = 0.0136;L0 = 0.07;
            b4 = 0.829;
            y2_0 = 30.2;
            deltaT = (y2_0 - Tmin)/b4;
            v = log(2/(PD-10^(-2)* x0));
            
            a1 = (L0^2 * ro * c) / (k * t_span);
            a2 = L0^2 * ro * c / k;
            a3 = ((ro*L0^2)/(k*deltaT))*beta*exp(v*x0);
            a4 = v * L0;
            % a5 = L0*h/k(i);
            c = a1;
            f = dudx;
            s = -wb * a2 * u + P*a3 * exp(-a4 * x);
    end

    % --------------------------------------------------------------------------
    function u0 = OneDimBHic(x)
            u0 = 0;
    end

    % --------------------------------------------------------------------------
    function [pl, ql, pr, qr] = OneDimBHbc(xl, ul, xr, ur, t)
            [~, y1, ~, ~] = ic_bc_src(xr, t);
            [~, ~, ~, y3] = ic_bc_src(xl, t);

            a5 = 175; % Heat transfer coefficient
            pl = -a5 * (-y3 + ul); % Robin bc
            ql = 1;

            pr = ur - y1;        % Dirichlet boundary condition: u(x=1, t) = y1
            qr = 0;
    end
end

function [T0] = extract_optimal_T0(t,u)
    myfittype = fittype('(1-exp(-t/T0))',...
                    'independent','t','dependent','u',...
                    'coefficients',{'T0'});
    fitresult = fit(t',u,myfittype,'StartPoint',0.1);
    T0 = fitresult.T0;
end
