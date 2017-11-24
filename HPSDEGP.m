classdef HPSDEGP < handle
    %CLUSTERINGCRWR learns multiple policies using clustering and
    % a contextual policy search algorithm
    
    properties
        
        ClusterSeedsNum = 10;
        PolicySet;
        Sdim;
        Adim;
        z_reward_laplace;
        ActionSet = [];
        ContextSet = [];
        RewardSet = [];
        SampleProbSet = [];
        FeatureSet = [];
        
        FeatureType = 1; % 0: linear, 1: exponential
        feature_h = 10;
        
        %Parameters for GP
        meanfunc = [];                    % empty: don't use a mean function
        covfunc = @covSEiso;              % Squared Exponental covariance function
        likfunc = @likGauss;              % Gaussian likelihood
        hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);
        hyp2;
    end
    
    methods
        function obj = HPSDEGP(SeedsNum, sdim, adim, UpdateType)
            obj.ClusterSeedsNum = SeedsNum;
            
            if strcmp( UpdateType, 'RWR'), obj.PolicySet = ContextualRWR( [ SeedsNum, 1 ] ); end
            if strcmp( UpdateType, 'REPS'), obj.PolicySet = ContextualREPS( [ SeedsNum, 1 ] ); end
            if ~strcmp( UpdateType, 'RWR') && ~strcmp( UpdateType, 'REPS')
                obj.PolicySet = ContextualRWR( [ SeedsNum, 1 ] ); 
            end
            
            obj.Sdim = sdim;
            obj.Adim = adim;
            
            %obj.gp = GaussianProcess;
        end
        
        function StoreSamples(obj, context, action,  reward)
            obj.ActionSet = [ obj.ActionSet;  action ];
            obj.ContextSet = [ obj. ContextSet; context ];
            obj.RewardSet = [ obj.RewardSet; reward];
        end
        
        function StoreSampleProb(obj, p)
            obj.SampleProbSet = [ obj.SampleProbSet; p ];
        end
        
        function SelectFeatureType(obj, type)
            obj.FeatureType = type;
            for i = 1: obj.ClusterSeedsNum
                obj.PolicySet( i, 1 ).SelectFeatureType( type ); 
            end
        end
        
        function SetContextSamples(obj, sampleNum)
            obj.ContextSamples = obj.ContextSet( 1:sampleNum, : );
        end
        
        function f = ComputeFeature(obj, context)
             switch obj.FeatureType
                case 0 %linear feature
                    f = LinearFeature( context );
                
                case 1 %exponential feature
                    if isempty( obj.ContextSamples )
                        disp( 'need to set ContextSamples. \n' );
                        obj.SetContextSamples( 5 );
                    end
                    f = ExponentialFeature( context, obj.ContextSamples, obj.feature_h );
            end
        end
        
        function OptimizeGPhyperParam(obj, SA, R)
            obj.hyp2 = minimize(obj.hyp, @gp, -100, @infGaussLik, obj.meanfunc, obj.covfunc, obj.likfunc, SA, R);
        end
        
        function [mu_r, var_r] = PredictReward( obj, SA, R, test_SA )
             [mu_r, var_r] = gp(obj.hyp2, @infGaussLik, obj.meanfunc, obj.covfunc, obj.likfunc, SA, R, test_SA);
        end
        
        function [action_new, pi_opt, p] = GenerateAction(obj, test_context)
            mean_gp = zeros( obj.ClusterSeedsNum, 1 );
            cov_gp = zeros( obj.ClusterSeedsNum, 1 );      
            Rmax = max( obj.RewardSet );
            sampleNum = 5;
            for i = 1:obj.ClusterSeedsNum
                 if isempty( obj.PolicySet( i, 1 ).policy_W )
                    mean_gp( i , 1 ) = - 1e+15;
                 else
                    for j =1:sampleNum
                        [action_i, ~] =  obj.PolicySet( i, 1 ).GenerateAction( test_context ); % generate the action
                        [mean, var ] = PredictReward( obj, [ obj.ContextSet, obj.ActionSet ], ... 
                                                      obj.RewardSet - Rmax, [ test_context, action_i ] ); % evaluate the action
                        mean_gp( i, 1) = mean_gp( i, 1) + mean / sampleNum;
                        cov_gp( i, 1) = cov_gp( i, 1) + var / sampleNum;
                    end
                end
            end
            
            UCB = mean_gp + 1*sqrt(cov_gp); % select the policy based on the UCB acquisition function
            [~,  pi_opt] = max(UCB);
            
            [action_new, p] = obj.PolicySet( pi_opt, 1 ).GenerateAction( test_context ); 
            
        end
        
         function [action_new, pi_opt] = GenerateGreedyAction(obj, test_context)
            mean_gp = zeros( obj.ClusterSeedsNum, 1 );
            cov_gp = zeros( obj.ClusterSeedsNum, 1 );
            Rmax = max(  obj.RewardSet );
            
            for i = 1:obj.ClusterSeedsNum
                 if isempty( obj.PolicySet( i, 1 ).policy_W )
                    mean_gp( i , 1 ) = - 1e+15;
                 else
                    action_i =  obj.PolicySet( i, 1 ).GenerateGreedyAction( test_context ); % generate the mean action
                    [mean_gp( i, 1), cov_gp(i, 1) ] = PredictReward( obj, [ obj.ContextSet, obj.ActionSet ], obj.RewardSet - Rmax, [ test_context, action_i ] ); % evaluate the action
                end
            end
            
            [~,  pi_opt] = max(mean_gp);
            action_new = obj.PolicySet( pi_opt, 1 ).GenerateGreedyAction( test_context ); 
            
        end
        
        function ClusterSamples(obj)
            % throw away samples with low returns
            n = size( obj.RewardSet, 1 );
            reward_sort = sort( obj.RewardSet );
            cut_n = round( 0.1*n );
            if n > 1250
                cut_n = n - 900;
            end
            
            reward_thre = reward_sort( cut_n  );
            e = ( obj.RewardSet >= reward_thre );
            ActionBatch = obj.ActionSet( e, :  );
            ContextBatch = obj.ContextSet( e, :  );
            RewardBatch = obj.RewardSet( e, :  );
            SampleProbBatch = obj.SampleProbSet( e, : );
            
            obj.ActionSet = ActionBatch;
            obj.ContextSet = ContextBatch;
            obj.RewardSet = RewardBatch;
            obj.SampleProbSet =SampleProbBatch;
             
            Rmax = max( RewardBatch );
            Rmin = min( RewardBatch );

            D = [ContextBatch, ActionBatch];
           
            [n, ~] = size(D);
            
            D_laplace = LaplacianEigenMapping(D, 25, 8 )';   %30, 10 / 20,5
            [~, n] = size( D_laplace );
            
            z_ini = mod( randperm(n), obj.ClusterSeedsNum ) + 1;
            iteration = 1000;
            weights = exp( 1.0* (RewardBatch - Rmax)/ (Rmax -Rmin) )' ;
            weights = weights ./ SampleProbBatch';
            weights = n * weights/sum(weights);

            disp('Start clustering...');
            t = cputime;
            for i = 1:5
                obj.z_reward_laplace = IWVBEMGMM(D_laplace, obj.ClusterSeedsNum, z_ini, weights, iteration);
                cluster = size( unique(  obj.z_reward_laplace  ), 2 );
                if cluster > 1
                    cluster
                    break;
                end
                
            end
            
            mess = [ 'computation time for clustering: ', sprintf( '%f', cputime - t ) ];
            disp( mess );
            
            
            for i = 1:obj.ClusterSeedsNum
                g = ( obj.z_reward_laplace == i ); 
                Si = ContextBatch( g , : );
                Ai = ActionBatch( g , : );
                Ri = RewardBatch( g , : );

                obj.PolicySet( i, 1 ).ContextSet = Si;
                obj.PolicySet( i, 1 ).ActionSet = Ai;
                obj.PolicySet( i, 1 ).RewardSet = Ri;
                obj.PolicySet( i, 1 ).ContextSamples = [];
            end
            
            disp('Clustering is done.');
            
        end
        
        function pi_opt = UpdatePolicySet(obj)
            for i = 1:obj.ClusterSeedsNum
                if size( obj.PolicySet( i, 1 ).ActionSet, 1 ) > obj.Adim
                    obj.PolicySet( i, 1 ).PolicyUpdate();
                else
                    obj.PolicySet( i, 1 ).policy_W = [];
                    obj.PolicySet( i, 1 ).policy_Cov = [];
                    obj.PolicySet( i, 1 ).ContextSamples = [];
                end
            end
            
            disp('Updated policies.');
            
            Rmax = max( obj.RewardSet );
            Rmin = min( obj.RewardSet );
            obj.OptimizeGPhyperParam( [obj.ContextSet, obj.ActionSet], (obj.RewardSet - Rmax) /(Rmax - Rmin) );
            
        end
        
        function policyNum = CheckOptionNum(obj )
            num = 0; 
            for i = 1:obj.ClusterSeedsNum
                if isempty( obj.PolicySet( i, 1 ).ActionSet )
                    
                else
                    num = num + 1;
                end
            end
            policyNum = num;
        end
    end
    
end

