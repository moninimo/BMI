function[predMSE,meanMSE]=mse_v_time_final(Xpred,Ypred,xt,yt,mx,my,in_data)
%xt,yt = test set hand position (
%mx,my = mean training hand position
%XYin = predicted values

ert=zeros(1,size(xt,2));
ert1=ert;
for k=1:8
    for n=1:size(Xpred,1)
        for ind=1:size(xt,2)
            if ind<=size(Xpred,2)
                predX=Xpred(n,ind,k);
                predY=Ypred(n,ind,k);
            else
                predX=Xpred(n,end,k);
                predY=Ypred(n,end,k);
            end
            if ind<=size(mx,2)
                predX_mean=mx(1,ind,k);
                predY_mean=my(1,ind,k);
            else
                predX_mean=mx(1,end,k);
                predY_mean=my(1,end,k);
            end
            ert(ind)=ert(ind)+((xt(n,ind,k)-predX).^2+(yt(n,ind,k)-predY).^2)*in_data(n,ind,k);
            ert1(ind)=ert1(ind)+((xt(n,ind,k)-predX_mean).^2+(yt(n,ind,k)-predY_mean).^2)*in_data(n,ind,k);
        end
    end
end
predMSE=ert/sum(sum(sum(in_data)));
meanMSE=ert1/sum(sum(sum(in_data)));
end