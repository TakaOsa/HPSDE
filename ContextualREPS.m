classdef ContextualREPS < handle
    %CONTEXTUALREPS performs contextual relative entropy policy search
    
    properties
        ActionSet = [];
        ContextSet = [];
        RewardSet = [];
        
        FeatureSet = [];
        featureType = 1;
        
        policy_W = [];
        policy_Cov = [];
        
        Adim;
        
        % parameters
        p_v;
        p_eta;
        
        FeatureType = 0;  % 0: linear feature, 1: exponential feature
        FeatureFunc;
        ContextSamples = [];
        feature_h = 1;
        KL_bound = 0.3;
    end
    
    methods
        function obj = ContextualREPS(F)
            if nargin ~= 0
                m = F(1);
                n = F(2);
                obj(m,n) = ContextualREPS;
            end
        end
        
        function StoreSamples(obj, context, action,  reward)
            obj.ActionSet = [ obj.ActionSet;  action ];
            obj.ContextSet = [ obj. ContextSet; context ];
            obj.RewardSet = [ obj.RewardSet; reward];
            
            obj.Adim = size(action, 2);
        end
        
         function SelectFeatureType(obj, type)
            obj.FeatureType = type;
        end
        
        function SetContextSamples(obj, sampleNum)
            n = size( obj.ActionSet, 1);
            if n > sampleNum
                obj.ContextSamples = obj.ContextSet( 1:sampleNum, : );
            else
                obj.ContextSamples = obj.ContextSet;
            end
        end
        
        function f = ComputeFeature(obj, context)
             switch obj.FeatureType
                case 0 %linear feature
                    f = LinearFeature( context );
                
                case 1 %exponential feature
                    if isempty( obj.ContextSamples )
                        disp( 'need to set ContextSamples. \n' );
                        if obj.Adim > 5
                            obj.SetContextSamples( obj.Adim * 2  );
                        else
                            obj.SetContextSamples( 4 );
                        end
                    end
                    f = ExponentialFeature( context, obj.ContextSamples, obj.feature_h );
                
            end
        end
        
        function PolicyUpdate(obj)
            [~, dim ]= size(obj.ActionSet);
            obj.Adim = dim;
            
            F = obj.ComputeFeature( obj.ContextSet );
            obj.FeatureSet = F;
            
            [~, fdim] = size( F );
            param_ini = 1 + rand( 1, fdim +1 );
            obj.OptimizeDualFunc( param_ini );
            
            bellmanErr = obj.RewardSet - F * obj.p_v ;
            maxBellmanErr = max( bellmanErr );
            
            d = exp( ( bellmanErr - maxBellmanErr )/ obj.p_eta  ) ;
            d = d / mean(d);
            
            D = diag( d );
            Z =  ( sum( d )^2 -  sum( d.^2) ) / sum(d);
            
            A = F' * D * F + 1e-3 * eye(fdim);
            obj.policy_W = A\  F' * D * obj.ActionSet;
            
            t = repmat( d, 1, dim ).*(obj.ActionSet -  F * obj.policy_W );
            
            if isempty( obj.policy_Cov ) &&  isnan(Z)
                obj.policy_Cov =  0.0001 * eye( dim, dim );
            elseif isempty( obj.policy_Cov ) &&  ~isnan(Z)
                 obj.policy_Cov = t' * t / Z;
            elseif isnan(Z)
                obj.policy_Cov = obj.policy_Cov;
            else
                obj.policy_Cov = 0.5* obj.policy_Cov + 0.5*t' * t / Z;
            end

            if unique( obj.policy_W ) == 0 
                obj.policy_W  = [];
                obj.policy_Cov = [];
            end
            
        end
        
        function SetKLbound(obj, epsilon)
            obj.KL_bound = epsilon;
        end
            
        function divKL = ComputeSampleBasedKL(obj)
            F = obj.FeatureSet;
            bellmanErr = obj.RewardSet - F * obj.p_v;
            maxBellmanErr = max( bellmanErr );
                        
            Z = sum( exp( (bellmanErr - maxBellmanErr)  / obj.p_eta) );
            p = exp(  (bellmanErr - maxBellmanErr)  / obj.p_eta ) / Z;
            
            divKL = sum( p .* log( p * numel(p) ) );            
        end
        
        function [f, g] = EvaluateDualFunc(obj, param) 
            pdim = numel(param);
            eta = param(1);
            v = param( 2:pdim );
            g = zeros( 1, pdim);
            
            if isempty(obj.FeatureSet) 
                disp( 'Compute Features before optimizing the dual function.' );
            end
            [n, d] = size( obj.FeatureSet );
            featureExpect = mean( obj.FeatureSet );        
            valueFuncExp = featureExpect*v;
            
            bellmanErr = obj.RewardSet - obj.FeatureSet * v;
            maxBellmanErr = max( bellmanErr );
            
            Z1 = sum( exp( (bellmanErr - maxBellmanErr )/ eta ) );
            Z2 = sum( (bellmanErr - maxBellmanErr ) .* exp( (bellmanErr - maxBellmanErr )/ eta ) );
            
            f = eta * obj.KL_bound + eta * log ( Z1 / n ) + maxBellmanErr + valueFuncExp;                        
            g(1) = obj.KL_bound + log(Z1 / n)  - Z2 / (eta * Z1 ) ;
            gv = sum( (repmat( featureExpect, n, 1) - obj.FeatureSet) .* repmat( exp( (bellmanErr - maxBellmanErr) / eta ), 1, d) )  / Z1; 
                        
            g(2:pdim) = gv;
            if g(1) < 0 && eta <= 0
                g( 1 ) = 0;
            end
            g = g';
            
        end
        
        function OptimizeDualFunc(obj, param_ini)
            param = param_ini;
            dim = numel( param_ini );
            if obj.FeatureType == 0
                param(dim) = 1;
            end
            
            LB = - inf(dim,1);
            LB(1) = 10^-8;
            UB = inf(dim,1);
            param_opt = minConf_TMP(@obj.EvaluateDualFunc, param', LB,UB);
                        
            obj.p_eta = param_opt(1);           
            obj.p_v = param_opt(2:dim);
                       
            test_KL = obj.ComputeSampleBasedKL();
            
            mess = [ 'resulting KL: ', sprintf('%f', test_KL), ' param_opt: ', sprintf(' %f ', param_opt  ) ];
            disp(mess);
            
        end
        
        function [action_new, p] = GenerateAction( obj, context_test )
            f =  obj.ComputeFeature( context_test  );
            mu =f * obj.policy_W;
            action_new = mvnrnd( mu, obj.policy_Cov );
            p = mvnpdf( action_new, mu, obj.policy_Cov );

        end
        
        function action_new = GenerateGreedyAction( obj, context_test )
            f =  obj.ComputeFeature( context_test  );
            action_new = f * obj.policy_W ;
        end
        
    end % end of method
    
end

