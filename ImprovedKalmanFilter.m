classdef ImprovedKalmanFilter < handle
    % Standard Kalman Filter with Position and Velocity State Space
    
    properties (Dependent)
        model
    end
    properties
        k           % time steps trained
        n_trials    % number of trials processed
        RMSE_history % history of RMSE values
        RMSe        % root mean squared error
        Xpred       % predicted states
        Xtrue       % true states
        bin_size    % bin size in milliseconds
        plotLC      % flag for plotting learning curve
    end
    properties (Access = private)
        A           % state transition matrix (nxn)
        H           % observation matrix (mxn)
        W           % process noise covariance   
        Q           % measurement covariance
        P           % estimate covariance
        K           % Kalman gain
        X           % state estimate
        alpha double % learning rate for system identification
        sysID logical % flag for system identification
    end

    methods
        function obj = ImprovedKalmanFilter(alpha, bin_size, args)
            arguments
                alpha double = 0.1
                bin_size double = 20
                args.plot logical = false
            end
            
            n_neurons = 98;
            n_states = 4;  % [x, y, vx, vy]
            
            obj.plotLC = args.plot;
            obj.k = 0;
            obj.sysID = false;
            obj.n_trials = 0;
            obj.bin_size = bin_size;

            % Initialize history tracking
            obj.RMSE_history = [];
            obj.RMSe = [];

            % Initialize state transition matrix with velocity model
            % [ x_t+1 ]   [ 1 0 dt 0  ] [ x_t  ]
            % [ y_t+1 ] = [ 0 1 0  dt ] [ y_t  ]
            % [ vx_t+1]   [ 0 0 1  0  ] [ vx_t ]
            % [ vy_t+1]   [ 0 0 0  1  ] [ vy_t ]
            dt = bin_size/1000; % convert bin size to seconds
            obj.A = [1 0 dt 0;
                     0 1 0 dt;
                     0 0 1 0;
                     0 0 0 1];
            
            obj.H = zeros(n_neurons, n_states);  
            obj.W = eye(n_states) * 0.01; 
            obj.Q = eye(n_neurons) * 0.1;
            obj.P = eye(n_states) * 0.1;
            obj.X = zeros(n_states, 1);      % Initial state estimate
            obj.K = zeros(n_states, n_neurons);

            obj.alpha = alpha;
        end

        function value = get.model(obj)
            mod.A = obj.A;               
            mod.H = obj.H;  
            mod.W = obj.W; 
            mod.Q = obj.Q; 
            mod.P = obj.P;
            mod.X = obj.X;
            mod.K = obj.K;
            value = mod;
        end

        function obj = fit(obj, Z_train, X_train)
            % Train Kalman Filter Decoder
            % Z_train: [n_neurons, n_samples] - Neural data
            % X_train: [n_outputs, n_samples] - Outputs to predict (positions only)
            
            % Derive velocities from positions if needed
            if size(X_train, 1) == 2
                X_full = obj.addVelocities(X_train);
            else
                X_full = X_train;
            end
            
            if ~obj.sysID
                obj.sysID = true;
                a = 1;
            else
                a = obj.alpha;
            end

            % Number of samples
            M = size(X_full, 2);
            
            % Current states and observations
            X0 = X_full;
            Z0 = Z_train;
            
            % States at time t and t+1
            X1 = X_full(:, 1:M-1);
            X2 = X_full(:, 2:M);

            % Calculate state transition matrix 
            calcA = X2 * X1' / (X1 * X1');
            obj.A = (1-a)*obj.A + a * calcA;

            % Calculate observation matrix
            calcH = Z0 * X0' / (X0 * X0');
            obj.H = (1-a)*obj.H + a * calcH;

            % Calculate process noise covariance 
            calcW = (X2 - obj.A * X1) * (X2 - obj.A * X1)' / (M-1) + 0.01*eye(size(obj.W));
            % calcW = (X2 - obj.A * X1) * (X2 - obj.A * X1)' / (M-1);
            obj.W = (1-a)*obj.W + a * calcW;

            % Calculate measurement noise covariance 
            calcQ = ((Z0 - obj.H * X0) * (Z0 - obj.H * X0)') / M;
            obj.Q = (1-a)*obj.Q + a * calcQ;
        end
        
        function full_state = addVelocities(obj, pos_data)
            % Add velocity states to position data
            % pos_data: [2, n_samples] - X,Y positions
            
            n_samples = size(pos_data, 2);
            full_state = zeros(4, n_samples);
            
            % Copy positions
            full_state(1:2, :) = pos_data;
            
            % Calculate velocities (forward difference)
            dt = obj.bin_size/1000; % convert to seconds
            
            % Initialize velocities to zero
            full_state(3:4, 1) = [0; 0];
            
            % Compute velocities for remaining time points
            for t = 2:n_samples
                full_state(3, t) = (pos_data(1, t) - pos_data(1, t-1)) / dt;
                full_state(4, t) = (pos_data(2, t) - pos_data(2, t-1)) / dt;
            end
        end

        function obj = predict(obj, Z_new, X_true_pos)
            % Initialize state with first position and zero velocity
            if size(X_true_pos, 1) == 2
                X_true_full = obj.addVelocities(X_true_pos);
            else
                X_true_full = X_true_pos;
            end
            
            obj.X = X_true_full(:, 1);
            obj.clearData();
            
            k_steps = size(Z_new, 2);

            for t = 1:k_steps
                obj.updateLoop(Z_new(:, t), X_true_full(:, t));     
            end
            
            obj.RMSe(end+1) = mean(obj.RMSE_history);
            obj.n_trials = obj.n_trials + 1;
        end

        function obj = updateLoop(obj, Z, X_true)
            % Recursive update step
            % Z: Neural activity at current timestep
            % X_true: True state (X, Y, Vx, Vy)

            % Time update %
            X_pred = obj.A * obj.X;                  % Predict state
            P_pred = obj.A * obj.P * obj.A' + obj.W; % Predict error covariance
        
            % Measurement update %
            % Kalman gain + a small regularization term
            innovation_cov = obj.H * P_pred * obj.H' + obj.Q + eye(size(obj.Q)) * 1e-6;
            obj.K = P_pred * obj.H' / innovation_cov;
            
            % Update estimate uncertainty
            obj.P = (eye(size(obj.P)) - obj.K * obj.H) * P_pred;
            
            % Update state estimate
            innovation = Z - obj.H * X_pred;
            obj.X = X_pred + obj.K * innovation;

            % Store predictions and true values
            obj.Xpred(:, end+1) = obj.X;
            obj.Xtrue(:, end+1) = X_true;
        
            % Compute RMSE for position only
            error = X_true(1:2) - obj.X(1:2);
            rmse = sqrt(mean(error.^2));
            obj.RMSE_history(end+1) = rmse;
            obj.k = obj.k + 1;
        end
        
        function [X_pos, Y_pos] = getPositionEstimate(obj)
            % Extract position estimates from the state vector
            X_pos = obj.X(1);
            Y_pos = obj.X(2);
        end
        
        function plotValues(obj)
            time = size(obj.Xtrue, 2);
            figure(1);
            clf(1)
            
            % Plot positions
            subplot(1, 3, 1);
            hold on;
            plot(1:time, obj.Xpred(1, :), 'b-', 'LineWidth', 1.5);
            plot(1:time, obj.Xpred(2, :), 'g-', 'LineWidth', 1.5);
            plot(1:time, obj.Xtrue(1, :), 'r--', 'LineWidth', 1.0);
            plot(1:time, obj.Xtrue(2, :), 'm--', 'LineWidth', 1.0);
            legend({'X pred', 'Y pred', 'X true', 'Y true'});
            title('Position Tracking');
            xlabel('Time bins');
            ylabel('Position (mm)');
            grid on;
            
            % Plot velocities
            subplot(1, 3, 2);
            hold on;
            plot(1:time, obj.Xpred(3, :), 'b-', 'LineWidth', 1.5);
            plot(1:time, obj.Xpred(4, :), 'g-', 'LineWidth', 1.5);
            plot(1:time, obj.Xtrue(3, :), 'r--', 'LineWidth', 1.0);
            plot(1:time, obj.Xtrue(4, :), 'm--', 'LineWidth', 1.0);
            legend({'Vx pred', 'Vy pred', 'Vx true', 'Vy true'});
            title('Velocity Tracking');
            xlabel('Time bins');
            ylabel('Velocity (mm/s)');
            grid on;
            
            % Plot RMSE
            subplot(1, 3, 3);
            plot(1:length(obj.RMSe), obj.RMSe, 'k-', 'LineWidth', 1.5);
            title('Position RMSE over Time');
            xlabel('Trials');
            ylabel('RMSE (mm)');
            grid on;
        end
        
        function obj = clearData(obj)
            obj.Xpred = [];
            obj.Xtrue = [];
            obj.RMSE_history = [];
        end
    end
end