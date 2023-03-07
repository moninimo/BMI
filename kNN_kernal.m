function [labels,err] = kNN_kernal(ord,test_feat,train_feat_mat,train_lab,NN_vec)
labels=zeros(length(NN_vec),size(test_feat,2));
err=zeros(length(NN_vec),size(test_feat,2));
for n=1:size(test_feat,2)
    test_feat_vec=test_feat(:,n);
    if ord==3
        aa=sqrt(sum(test_feat_vec.^2));%scalar
        ab=sqrt(sum(train_feat_mat.^2));%row vector
        [er1,ind1]=maxk(test_feat_vec'*train_feat_mat./(aa*ab),max(NN_vec));
    else
        [er1,ind1]=mink(sum((abs(test_feat_vec-train_feat_mat)).^ord),max(NN_vec));
    end
    for ind_NN=1:length(NN_vec)
        NN=NN_vec(ind_NN);
        ind=ind1(1:NN);
        er=er1(1:NN);
        train_lab1=train_lab(ind);
        [~,~,temp]=mode(train_lab1);
        er=er(ismember(train_lab1,temp{1}'));
        train_lab1=train_lab1(ismember(train_lab1,temp{1}'));
        [temper,temp]=min(er);
        labels(ind_NN,n)=train_lab1(temp);
        err(ind_NN,n)=temper;
    end
end
end





