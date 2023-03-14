function [sb,sw]=make_sb_sw(X,l)
c=unique(l);
mc=zeros(size(X,1),length(c));
mx=mean(X,2);
for im_id=1:length(c)
    mc(:,im_id)=mean(X(:,l==c(im_id)),2);
end
sb=(mc-mx)*(mc-mx)';
st=(X-mx)*(X-mx)';
sw=st-sb;
end