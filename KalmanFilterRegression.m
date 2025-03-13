classdef KalmanFilterRegression < handle
    % Class for the Kalman Filter Decoder
    properties (Dependent)
        model
    end
    properties
        kloop    % time steps trained
        n_trials
        Kx
        Ky
        RMSE_history
        RMSe
        Xpred
        
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
        n_states
        delayStp
        Xtrue
        measurementBuffer
    end

    methods
        function obj = KalmanFilterRegression(args)
            arguments
                args.alpha double = 0.1;
                args.n_neurons double = 98;
                args.binSize double = 10;  %in ms
                args.delaySteps double = 0;
            end
            n_neurons = args.n_neurons;
            n_states = 6;  % disp,vel,accel
            obj.n_states = n_states;
            obj.kloop = 0; 
            obj.sysID = false;
            obj.n_trials = 0;

            % Initialize delay buffer
            obj.delayStp = args.delaySteps;
            obj.measurementBuffer = [];

            % Initialize history tracking
            obj.Kx = [];
            obj.Ky = [];
            obj.RMSE_history = [];
            obj.RMSe = [];

             % Initialize state transition matrix with velocity and acceleration model
            dt = args.binSize/1000; % convert bin size to seconds
            dt2 = 0.5*dt^2;
            obj.A = [1 0 dt 0  dt2 0;
                     0 1 0  dt 0   dt2;
                     0 0 1  0  dt  0;
                     0 0 0  1  0   dt;
                     0 0 0  0  1   0;
                     0 0 0  0  0   1];               
            obj.H = zeros(n_neurons,n_states);  
            obj.W = eye(n_states) * 0.01; 
            obj.Q = eye(n_neurons) * 0.1;
            obj.P = eye(n_states) * 0.1;
            obj.X = zeros(n_states,1);      % Initial state estimate
            obj.K = zeros(n_states,n_neurons);

            obj.alpha = args.alpha;
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
            % if ~obj.sysID
            %     obj.sysID = true;
            %     a = 1;
            % else
            %     a = obj.alpha;
            % end

            a = obj.alpha;

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
            % obj.X = X_train(:,1);

            obj.Xpred = [];
            obj.Xtrue = [];
            obj.RMSE_history = [];

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
            obj.measurementBuffer = [obj.measurementBuffer, Z];
            regInv = eye(size(obj.Q)) * 1e-6;

            % Check if enough measurements have been buffered
            if size(obj.measurementBuffer,2) > obj.delayStp
                % Retrieve delayed measurement
                Z_delayed = obj.measurementBuffer(:, 1);
                
                % Remove used measurement from buffer
                obj.measurementBuffer(:, 1) = [];
            else
                % If buffer is not full, skip update
                Z_delayed = zeros(size(Z));
            end

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
            obj.X = X_pred + obj.K * (Z_delayed - obj.H * X_pred);

            % Save the Position states only
            obj.Xpred(:,end+1) = obj.X(1:2,:);
            obj.Xtrue(:,end+1) = X_true(1:2,:);
        
            % Compute RMSE for position only
            error = X_true(1:2,:) - obj.X(1:2,:);
            rmse = sqrt(mean(error.^2));
            obj.RMSE_history(end+1) = rmse;
            obj.kloop = obj.kloop + 1;
        end
        function plotValues(obj,RMSEperTrial)
            arguments
                obj
                RMSEperTrial logical = true;
            end
            time = size(obj.Xtrue,2);
            figure(1);
            clf(1);
            subplot(1,2,1);
            hold on
            plot(1:time,obj.Xpred,LineWidth=1.5,Color='b');
            hold on
            plot(1:time,obj.Xtrue,'LineStyle','--',LineWidth=1.0,Color='r');
            xlabel('time bins / window')
            ylabel('XY positions / cm')
            legend({'Xpred','Ypred','Xreal','Yreal'},Location="northwest");

            title('Real vs Predicted values');

            subplot(1,2,2);
            hold on
            if RMSEperTrial
                rms_t = 1:length(obj.RMSe);
                plot(rms_t,obj.RMSe,LineWidth=1.0,Color='g');
                title('RMSe per trial');
                xlabel('Trial number');
            else
                plot(1:time,obj.RMSE_history,'LineStyle','-',LineWidth=1.0,Color='g');
                title('RMSe in trial');
                xlabel('time bins / window')
            end
            ylabel('RMSe /cm')

        end
        function obj = clearRMSe(obj)
            obj.RMSe = [];
        end
        function obj = setInitialPos(obj,XYpos)
            obj.X = cat(1,XYpos,zeros(obj.n_states-2,1));
        end
    end
end