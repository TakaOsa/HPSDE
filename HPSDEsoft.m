classdef HPSDEsoft < handle
    %CLUSTERINGCRWR learns multiple policies using clustering and
    % a contextual policy search algorithm
    
    properties
        
        ClusterSeedsNum = 10;
        OptionNum;
        PolicySet;
        Sdim;
        Adim;
        z_reward_laplace;
        ActionSet = [];
        ContextSet = [];
        RewardSet = [];
        SampleProbSet = [];
        OptionFeatureType = 0;
        
        cluster_w =[];
        cluster_W =[];
        cluster_h =[];
        cluster_nu = [];
        cluster_a = [];
        
        gatePolicy;
        GateFeatureSet = [];
        GateFeatureType = 1; % 0: linear feature, 1: exponential feature
        ContextSamples = [];
        feature_h = 1;
    end
    
    methods
        function obj = HPSDEsoft(SeedsNum, sdim, adim, UpdateType)
            obj.ClusterSeedsNum = SeedsNum;
            
            if strcmp( UpdateType, 'RWR'), obj.PolicySet = ContextualRWR( [ SeedsNum, 1 ] ); end
            if strcmp( UpdateType, 'REPS'), obj.PolicySet = ContextualREPS( [ SeedsNum, 1 ] ); end
            if ~strcmp( UpdateType, 'RWR') && ~strcmp( UpdateType, 'REPS')
                obj.PolicySet = ContextualRWR( [ SeedsNum, 1 ] ); 
            end
            
            obj.Sdim = sdim;
            obj.Adim = adim;
            obj.gatePolicy = SoftMaxRegression(  );
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
            obj.OptionFeatureType = type;
            for i = 1: obj.ClusterSeedsNum
                obj.PolicySet( i, 1 ).SelectFeatureType( type ); 
            end
        end
        
        function po = ComputeOptionProb(obj, context)
            po = zeros( obj.ClusterSeedsNum, 1 );
            p_so = zeros( obj.ClusterSeedsNum, 1 );
            
            for o = 1:obj.ClusterSeedsNum
                if ~isempty( obj.PolicySet( o, 1 ).policy_W ) 
                    W_o = obj.cluster_W( obj.Adim+1:obj.Adim+obj.Sdim, obj.Adim+1:obj.Adim+obj.Sdim, o);
                    mu_o = obj.cluster_h( obj.Adim+1:obj.Adim+obj.Sdim, o);
                    t = context - mu_o;
                    p_so( o ) = 1 / norm( inv(W_o) )^(0.5) * exp( -0.5* t' * W_o * t ) / (2 * pi)^( obj.Sdim/2 );
                end
            end
            po = obj.cluster_w' .* p_so / sum( obj.cluster_w );           
            po = po / sum( po );
        end
        
         function SelectGateFeatureType(obj, type)
            obj.GateFeatureType = type;
         end
        
         function f = ComputeFeature(obj, context, FeatureType)         
             switch FeatureType
                case 0 %linear feature
                    f = LinearFeature( context );

                case 1 %exponential feature
                    if isempty( obj.ContextSamples )
                        disp( 'need to set ContextSamples. \n' );
                        obj.SetContextSamples( 10 );
                    end
                    f = ExponentialFeature( context, obj.ContextSamples, obj.feature_h );
                    
                 case 2 % Quadratic feature
                    f = QuadraticFeature( context );
            end
         end
         
         function SetContextSamples(obj, sampleNum)
            obj.ContextSamples = obj.ContextSet( 1:sampleNum, : );
         end
         
         function UpdateGatePolicy(obj)
             n = size( obj.ActionSet, 1 );
             y = zeros( n, obj.OptionNum );
             for o = 1:obj.OptionNum
                y( obj.z_reward_laplace == o, o ) = 1;
             end
             obj.GateFeatureSet = obj.ComputeFeature( obj.ContextSet, obj.GateFeatureType );
             obj.gatePolicy.SetData( obj.GateFeatureSet, y );
             obj.gatePolicy.OptimizeTheta();
         end
         
         
        function [action_new, pi_opt, p] = GenerateAction(obj, test_context)
           f_gate = obj.ComputeFeature( test_context, obj.GateFeatureType );
           [po, class_test ] = obj.gatePolicy.Prediction( f_gate );
           for o = 1:obj.OptionNum
               if isempty(obj.PolicySet( o, 1 ).policy_W)
                   po( o ) = 0;
               end
           end
           cumsum_po = cumsum( po );
           pi_opt = find( cumsum_po > rand, 1 );
           %[~, pi_opt] = max(po, [], 2);
           
           [action_new, p_option] = obj.PolicySet( pi_opt, 1 ).GenerateAction( test_context ); 
           p = p_option * po( pi_opt );
        end
        
        function [action_new, pi_opt] = GenerateGreedyAction(obj, test_context)
            f_gate = obj.ComputeFeature( test_context, obj.GateFeatureType );
            [po, class_test ] = obj.gatePolicy.Prediction( f_gate );
            for o = 1:obj.OptionNum
                if isempty(obj.PolicySet( o, 1 ).policy_W)
                    po( o ) = 0;
                end
            end
           
            [~, pi_opt] = max(po, [], 2);
            
            action_new = obj.PolicySet( pi_opt, 1 ).GenerateGreedyAction( test_context ); 
        end
        
        function ClusterSamples(obj)
            n = size( obj.RewardSet, 1 );
            reward_sort = sort( obj.RewardSet );
            cut_n = round( 0.1*n );
            if n > 1250
                cut_n = n - 1000;
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
            [ n, ~ ] = size(D);
            D_laplace = LaplacianEigenMapping(D, 25, 8 )';        
            [~, n] = size( D_laplace );
            obj.z_reward_laplace = mod( randperm(n), obj.ClusterSeedsNum ) + 1;
            iteration = 1000;
            weights = exp( 10.0* (RewardBatch - Rmax)/ (Rmax -Rmin) )' ;
            weights = weights ./ SampleProbBatch';
            weights = n * weights/sum(weights);

            disp('Start clustering...');
            t = cputime; 

             obj.z_reward_laplace = IWVBEMGMM(D_laplace, obj.ClusterSeedsNum, obj.z_reward_laplace, weights, iteration);
            mess = [ 'computation time for clustering: ', sprintf( '%f', cputime - t ) ];
            disp( mess );
            
            obj.OptionNum = size( unique( obj.z_reward_laplace ), 2)  ;
            
            for i = 1:obj.ClusterSeedsNum
                g = ( obj.z_reward_laplace == i ); 
                Si = ContextBatch( g , : );
                Ai = ActionBatch( g , : );
                Ri = RewardBatch( g , : );

                obj.PolicySet( i, 1 ).ContextSet = Si;
                obj.PolicySet( i, 1 ).ActionSet = Ai;
                obj.PolicySet( i, 1 ).RewardSet = Ri;
            end
            
            disp('Clustering is done.');
            
            obj.UpdateGatePolicy();
            
        end
        
        function UpdatePolicySet(obj)
          
            for i = 1:obj.ClusterSeedsNum
                obj.PolicySet( i, 1 ).SelectFeatureType( obj.OptionFeatureType ); 
                
                if size( obj.PolicySet( i, 1 ).ActionSet, 1 ) > 5
                    obj.PolicySet( i, 1 ).PolicyUpdate();
                else
                    obj.PolicySet( i, 1 ).policy_W = [];
                    obj.PolicySet( i, 1 ).policy_Cov = [];
                end
            end
            
            disp('Updated policies.');
        end
        
        function policyNum = CheckOptionNum(obj )
            num = 0; 
            for i = 1:obj.ClusterSeedsNum
                if isempty( obj.PolicySet( i, 1 ).policy_W )
                    
                else
                    num = num + 1;
                end
            end
            policyNum = num;
        end
    end
    
end

