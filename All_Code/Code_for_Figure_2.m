clc;clear;close all
load 'monkeydata_training.mat'
col = get(gca, 'ColorOrder');
close
col=[col;[0.5 0.5 0.5]];
index=0;
figure(1)
M_lda=2;
M_pca=170;
k_vec=1:8;
res=500;
dt=80;
T_end=320;
ang=[30:40:230 310 350];
[F,l,t,mF]=organize_data1(trial,dt,T_end);
% [N,mx,U,L]=eigenmodel(F,2);
% W=U'*(F-mx);
[W]=PCA_LDA(F,l,M_pca,M_lda,[]);
for k=k_vec
    plot(W(1,l==k),W(2,l==k),'o','Color',col(k,:),'MarkerFaceColor',col(k,:))
    hold on
end
% for k=k_vec
%     for n=1:size(V,2)
%         x1=[V(1,n,k) Vm(1,1,k)];x1=x1(1):((x1(2)-x1(1))/10):x1(2);
%         y1=[V(2,n,k) Vm(2,1,k)];y1=y1(1):((y1(2)-y1(1))/10):y1(2);
%         plot(x1,y1,'Color',col(k,:));
%     end
% end
%     
%     
A_vec=[];
for k=1:8
    [A,C_1,m]=get_gauss_params(W(:,l==k));
    A_vec=[A_vec A];
end
map=exp(max(A_vec));
tic
for k=1:8
    k
    [A,C_1,m]=get_gauss_params(W(:,l==k));
    Y=W(:,l==k)-m;
    min_x=min(W(1,:));
    min_y=min(W(2,:));
    max_x=max(W(1,:));
    max_y=max(W(2,:));
    
    x_ax=linspace(min_x,max_x,res);
    x_ax_tot=[];
    y_ax_tot=[];
    p=[];
    for y1=linspace(min_y,max_y,res)
        Y=[x_ax;y1*ones(size(x_ax))];
        temp1=diag(Y'*C_1*Y);
        p1=exp(A-0.5*temp1);
        p=[p;p1];
        x_ax_tot=[x_ax_tot;x_ax'];
        y_ax_tot=[y_ax_tot;y1*ones(size(x_ax))'];
    end
    % hold off
    for cont=[1 4 16 64]*0.005
        ind=find(p>(cont*map));
        x_ax=x_ax_tot(ind);
        y_ax=y_ax_tot(ind);
        b1=boundary(x_ax,y_ax);
        plot(x_ax(b1)+m(1),y_ax(b1)+m(2),'k','LineWidth',1.5)
        plot(x_ax(b1)+m(1),y_ax(b1)+m(2),'Color',col(k,:),'LineWidth',1.5)
        hold on
    end
end
toc
grid on
grid minor
axis square
leg=legend(num2str(ang'));
leg.NumColumns=4;leg.FontSize=12;
xticklabels([]);
yticklabels([]);
title("Feature Vectors with \deltat=80ms, at t=320ms."+newline+"Transfromed using PCA-LDA with Mpca=170, Mlda=2",'FontSize',14)
% end
%%
function [W, Wt]=PCA_LDA(X,l,M_pca,M_lda,Xt)
[N,mx,Wpca,L]=eigenmodel(X,size(X,2));
if M_lda<length(unique(l)) && M_lda>0
    [sb,sw]=make_sb_sw(X,l);
    Wopt=make_Wopt(Wpca,M_pca,M_lda,sb,sw);
else
    Wopt=Wpca(:,1:M_pca);
end
W=Wopt'*(X-mx);
if ~isempty(Xt)
    Wt=Wopt'*(Xt-mx);
else
    Wt=[];
end
end