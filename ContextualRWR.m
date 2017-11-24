classdef ContextualRWR <handle
    %CONTEXTUALRWR class for Contextual Reward Weighted Regression 

    properties
        ActionSet = [];
        ContextSet = [];
        RewardSet = [];
        
        featureType = 1;
        
        Adim;
        
        policy_W = [];
        policy_Cov = [];
        
        FeatureType = 0;  % 0: linear feature, 1: exponential feature
        FeatureFunc;
        ContextSamples = [];
        feature_h = 1;
    end
    
    methods
        function obj = ContextualRWR(F)
            if nargin ~= 0
                m = F(1);
                n = F(2);
                obj(m,n) = ContextualRWR;
            end
        end
        
        function InitParam(obj)
            
        end
        
        function StoreSamples(obj, context, action,  reward)
            obj.ActionSet = [ obj.ActionSet;  action ];
            obj.ContextSet = [ obj. ContextSet; context ];
            obj.RewardSet = [ obj.RewardSet; reward];
            
            obj.Adim = size(action, 2);
        end
        
        function SetContextSamples(obj, sampleNum)
            n = size( obj.ActionSet, 1);
            if n > sampleNum
                obj.ContextSamples = obj.ContextSet( 1:sampleNum, : );
            else
                obj.ContextSamples = obj.ContextSet;
            end
        end
        
        function SelectFeatureType(obj, type)
            obj.FeatureType = type;
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
            Rmax = max( obj.RewardSet );
            Rmin = min( obj.RewardSet );
            [~, obj.Adim ]= size(obj.ActionSet);
            dim = obj.Adim;
            
            d = exp( 10 * (obj.RewardSet - Rmax)/(Rmax - Rmin) );
            d = d / mean(d);
            D = diag( d );
            F = obj.ComputeFeature( obj.ContextSet );
            
            Z =  ( sum( d )^2 -  sum( d.^2) ) / sum(d);
                       
            [~, fdim] = size( F );
            A = F' * D * F + 1e-3 * eye(fdim);
            obj.policy_W = A\  F' * D * obj.ActionSet;
            t = repmat( d, 1, dim ).*(obj.ActionSet -  F * obj.policy_W );
            
            if isempty( obj.policy_Cov ) &&  isnan(Z)
                obj.policy_Cov =  0.1 * eye( dim, dim );
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
        
        function [action_new, p] = GenerateAction( obj, context_test )
            f =  obj.ComputeFeature( context_test  );
            mu =obj.policy_W' * f';
            action_new = mvnrnd( mu, obj.policy_Cov );
            p = mvnpdf( action_new', mu, obj.policy_Cov );
        end
        
        function action_new = GenerateGreedyAction( obj, context_test )
            f =  obj.ComputeFeature( context_test  );
            action_new = f * obj.policy_W ;
        end
    end
    
end

