function [N,mx,U,L]=eigenmodel(x,p)
N=size(x,2);
mx=mean(x,2);
A=x-mx;
S=A'*A/N;
[U,L]=eig(S);
p=min(p,size(U,2));
[~,ind]=maxk(diag(L),p);
U=A*U(:,ind);
U=U./sqrt(sum(U.^2));
L=L(ind,ind);
end