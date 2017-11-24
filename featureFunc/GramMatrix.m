function [ G ] = GramMatrix( X, h )
%GRAMMATRIX 
% h: band width for the Gaussian kernel
% X contains samples in each row.

A = sum(X .* X, 2);
B = -2 * X * X';
K = bsxfun(@plus, A, B);
K = bsxfun(@plus, K, A');
G = exp( - K / h );

end

