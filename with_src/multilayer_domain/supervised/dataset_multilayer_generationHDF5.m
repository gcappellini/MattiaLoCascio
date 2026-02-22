N_samples = 100;
x_change  = linspace(0.2, 0.8, N_samples);

base_dir = "path/data";
if ~exist(base_dir,'dir')
    mkdir(base_dir);
end

h5file = fullfile(base_dir, 'dataset_bioheat.h5');

if exist(h5file,'file')
    delete(h5file);
end

for k = 1:N_samples

    x_interface = x_change(k);

    fprintf('Running simulation %d / %d (x_interface = %.3f)\n', ...
            k, N_samples, x_interface);

    % --- solver ---
    [T, x_grid, tlist] = OneDimBH_src_multi_layer(x_interface);

    % --- HDF5 iitalization---
    if k == 1
        Nt = length(tlist);
        Nx = length(x_grid);

        % main datasets
        h5create(h5file, '/T_data',   [Nt, Nx, N_samples], 'Datatype','double');
        h5create(h5file, '/x_change', [N_samples],         'Datatype','double');
        h5create(h5file, '/x_grid',   [Nx],                'Datatype','double');
        h5create(h5file, '/tlist',    [Nt],                'Datatype','double');

        % axis writing
        h5write(h5file, '/x_grid', x_grid);
        h5write(h5file, '/tlist',  tlist);
        h5write(h5file, '/x_change', x_change);
    end

    % --- solution ---
    h5write(h5file, '/T_data', T, [1,1,k], [Nt,Nx,1]);

end
