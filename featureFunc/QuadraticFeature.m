function [ f ] = QuadraticFeature( s )
%QUADRATICFEATURE 
% s need to be a row vector, e.g., n x1

    A = s * s';
    f = [diag(A); nonzeros( triu(A, 1) )  ];

end

