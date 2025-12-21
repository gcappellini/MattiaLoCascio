function [sol] = OneDimBH_src_multi_layer()

  m = 0;
  x = linspace(0,1,101);
  t = linspace(0,1,101);

  % Run PDE solver with nested functions
  sol = pdepe(m, @OneDimBHpde, @OneDimBHic, @OneDimBHbc, x, t);

  u1 = sol(:,:,1); % solution of system

  % Write solution to file
  fileID = fopen("C:\Users\matti\OneDrive\Desktop\pc matty\UNIVERSITA'\TESI Magistrale/gt_bioheat1D_src.csv", 'w');
  for i = 1:101
      for j = 1:101
          fprintf(fileID,'%6.2f %6.2f %12.8f\n', x(j), t(i), u1(i,j));
      end
  end
  fclose(fileID);

% Extract solution at a specific x point within the domain
x_sel = (length(x)+1)/2; % Middle point
x_meas = x(x_sel);
u_meas = u1(:,x_sel);
u_meas_1 = u1(:,17); % x=0.16

% Plot the solution u1 vs x and t
fig = figure();
surf(x, t, u1, 'EdgeColor', 'none');
xlabel('x');
ylabel('t');
zlabel('u1(x,t)');
beta = 1;
P =100;
title(sprintf('Solution of 1D Bioheat Equation (beta = %.2f), (Power = %.2f)', beta, P));
colorbar;
saveas(fig, 'bioheat_1D_src.png');

% Estimate optimal T0 from u_meas data
T0_optimal = extract_optimal_T0(t, u_meas);
fprintf('Estimated optimal T0 from u_meas: %.4f\n', T0_optimal);
T0_optimal1 = extract_optimal_T0(t, u_meas_1);
fprintf('Estimated optimal T0 from u_meas: %.4f\n', T0_optimal1);

% Plot u_meas vs t
fileID_u = fopen("C:\Users\matti\OneDrive\Desktop\pc matty\UNIVERSITA'\TESI Magistrale/u_meas_src.csv", 'w');
  for i = 1:101
    fprintf(fileID_u,'%6.2f %6.2f %12.8f\n', t(i), u1(i,x_sel)); 
  end
  fclose(fileID_u);

fig2 = figure();
plot(t, u_meas, 'b-', 'LineWidth', 2);
hold on;
u_fit = 1 - exp(-t / T0_optimal);
plot(t, u_fit, 'r--', 'LineWidth', 2);
legend('u_{meas}(t)', 'u_{fit}(t)', 'Location', 'Best');
xlabel('t');
ylabel(sprintf('u1(x=%.2f,t)', x_meas));
title(sprintf('Solution of 1D Bioheat Equation at x=%.2f, with T0=%2f', x_meas,T0_optimal));
grid on;
saveas(fig2, 'u_meas_src.png');


   % --------------------------------------------------------------------------
    function [c, f, s] = OneDimBHpde(x, t, u, dudx)
            wb = 0.0005;  % Uses config from outer scope

            %Parameters for multi-layer model
            beta = 1;
            P = 25;
            h = 525.0; t_span = 1800.035;
            c = [2348,3421]; ro = [911,1090]; k = [0.21,0.49]; % fat - muscle
            Tmin = 21.5; x0 = 0.004; PD = 0.0136;L0 = 0.07;
            b4 = 0.829;
            y2_0 = 30.2;
            deltaT = (y2_0 - Tmin)/b4;
            v = log(2/(PD-10^(-2)* x0));
            n_layers = 2;
            if x <= 1/n_layers
                i = 1;
            else
                i = 2;
            end
            a1 = (L0^2 * ro(i) * c(i)) / (k(i) * t_span);
            a2 = L0^2 * ro(i) * c(i) / k(i);
            a3 = ((ro(i)*L0^2)/(k(i)*deltaT))*beta*exp(v*x0);
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
    fitresult = fit(t',u,myfittype,'StartPoint',0.5);
    T0 = fitresult.T0;
end
