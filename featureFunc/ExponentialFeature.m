function [ f ] = ExponentialFeature( s, sampleSet, h )
%EXPONETIALFEATURE computes the exponential feature
%
% h: band width
% s: given state/context; s should be the column vector
% sampleSet: the dataset for computing the exponential feature
%                   sampleSet should be M x N where M is the number of the samples and N is the dimension of the state/context 

[n  d] = size( s );
[m d] = size( sampleSet);

S2 = sum( s.^2, 2 );
Set2 = sum( sampleSet .^2, 2 );
A = repmat( S2,  1, m) + repmat( Set2', n, 1  ) - 2* s * sampleSet'; 
hh = 2*h*h;
f = exp( - A / hh );

end

