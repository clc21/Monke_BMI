classdef KalmanFilterRegression < handle
    % Class for the Kalman Filter Decoder
    properties (Dependent)
        model
    end
    properties
        k    % time steps trained
        n_trials
        Kx
        Ky
        RMSE_history
        RMSe
        Xpred
        Xtrue
        plotLC
    end
    properties (Access = private)
        A   % state transition matrix (nxn)
        H   % state-to-measurement/observation matrix (mxn)
        W   % process noise covariance   
        Q   % measurement covariance
        P   % estimate covariance
        K   % Kalman gain
        X   % state estimate
        alpha double    % learning rate for system identification
        sysID logical
    end

    methods
        function obj = KalmanFilterRegression(alpha,args)
            arguments
                alpha double = 0.1;
                args.plot logical = false;
            end
            n_neurons = 98;
            n_states = 2;  % disp,vel,accel
            obj.plotLC = args.plot;
            obj.k = 0;
            obj.sysID = false;
            obj.n_trials = 0;

            % Initialize history tracking
            obj.Kx = [];
            obj.Ky = [];
            obj.RMSE_history = [];
            obj.RMSe = [];

            obj.A = zeros(n_states);               
            obj.H = zeros(n_neurons,n_states);  
            obj.W = zeros(n_states,n_states); 
            obj.Q = zeros(n_neurons,n_neurons); 
            obj.P = eye(n_states) * 0.1;
            obj.X = zeros(n_states,1);      % Initial state estimate
            obj.K = zeros(n_states,n_neurons);

            obj.alpha = alpha;
        end

        function value = get.model(obj)
            mod.A = obj.A;               
            mod.H = obj.H;  
            mod.W = obj.W; 
            mod.Q = obj.Q; 
            mod.P = obj.P;
            mod.X = obj.X;      % Initial state estimate
            mod.K = obj.K;
            value = mod;

        end

        function obj = fit(obj, Z_train, X_train)
            % Train Kalman Filter Decoder [row,col]
            % Z_train: [n_neurons,n_samples] - Neural data
            % X_train: [n_outputs,n_samples] - Outputs to predict
            if ~obj.sysID
                obj.sysID = true;
                a = 1;
            else
                a = obj.alpha;
            end

            M = size(X_train,2);  % n_samples

            X0 = X_train;
            Z0 = Z_train;
            X1 = X_train(:,1:M-1);
            X2 = X_train(:,2:M);

            calcA = X2 * X1' / (X1 * X1');
            obj.A = (1-a)*obj.A + a * calcA;

            calcH = Z0 * X0' / (X0 * X0');
            obj.H = (1-a)*obj.H + a * calcH;

            calcW = (X2 - obj.A * X1) * (X2 - obj.A * X1)' / (M-1);
            obj.W = (1-a)*obj.W + a * calcW;

            calcQ = ((Z0 - obj.H * X0) * (Z0 - obj.H * X0)' )/ M;
            obj.Q = (1-a)*obj.Q + a * calcQ;
        end

        function obj = predict(obj,Z_train,X_train)
            obj.X = X_train(:,1);
            obj.clearLCData();
            k_steps = size(X_train, 2);

            for t = 1:k_steps
                obj.updateLoop(Z_train(:, t), X_train(:, t));     
            end
            obj.RMSe(end+1) = mean(obj.RMSE_history);
            obj.n_trials = obj.n_trials+1;

            % obj.Kx(:,end+1) = obj.K(1,:)';
            % obj.Ky(:,end+1) = obj.K(2,:)';
        end

        function obj = updateLoop(obj, Z, X_true)
            % Recursive update step
            % Z: Neural activity (spike counts) at current timestep
            % X_true: True state (X and Y pos)

            regInv = eye(size(obj.Q)) * 1e-6;

            % Time update %
            % ----------- %
            X_pred = obj.A * obj.X;                  % Predict state
            P_pred = obj.A * obj.P * obj.A' + obj.W; % Predict error covariance
        
            % Measurement update %
            % ------------------ %
            % Kalman gain
            obj.K = P_pred * obj.H' / (obj.H * P_pred * obj.H' + obj.Q + regInv); 
            % Update estimate uncertainty
            obj.P = (eye(size(obj.P)) - obj.K * obj.H) * P_pred;
            % Update state estimate
            obj.X = X_pred + obj.K * (Z - obj.H * X_pred);

            obj.Xpred(:,end+1) = obj.X;
            obj.Xtrue(:,end+1) = X_true;
        
            % Compute RMSE
            error = X_true - obj.X;
            rmse = sqrt(mean(error.^2));
            obj.RMSE_history(end+1) = rmse;
            obj.k = obj.k + 1;
        end
        function plotValues(obj)
            time = size(obj.Xtrue,2);
            figure(1);
            clf(1);
            subplot(1,2,1);
            hold on
            plot(1:time,obj.Xpred,LineWidth=1.0,Color='b');
            hold on
            plot(1:time,obj.Xtrue,'LineStyle','--',LineWidth=2.0,Color='r');
            legend({'Xpred','Ypred','Xreal','Yreal'});
            title('Real vs Predicted values');

            subplot(1,2,2);
            rms_t = 1:length(obj.RMSe);
            hold on
            plot(rms_t,obj.RMSe,LineWidth=1.0,Color='g');
            title('RMSe per trial');
        end
        function obj = clearLCData(obj)
            obj.Xpred = [];
            obj.Xtrue = [];
            obj.RMSE_history = [];
        end
        function obj = clearK(obj)
            obj.Kx = [];
            obj.Ky = [];
        end
    end
end
