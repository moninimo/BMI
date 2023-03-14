function [Wopt]=make_Wopt(Wpca,M_pca,M_lda,sb,sw)
[Wlda,L] = eig((Wpca(:,1:M_pca)'*sw*Wpca(:,1:M_pca))^-1*Wpca(:,1:M_pca)'*sb*Wpca(:,1:M_pca));
[~,ind]=maxk(diag(L),M_lda);
Wopt=Wpca(:,1:M_pca)*Wlda(:,ind);
end