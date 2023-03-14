function [norm_matrix,coef1,coef2]=normalise_matrix(feature_mtx,a,b)    
    mtx_min = min(min(feature_mtx));
    mtx_max = max(max(feature_mtx));
    
    coef1 = (b-a)/(mtx_max-mtx_min);
    coef2 = -((b-a)*mtx_min/(mtx_max-mtx_min)) + a;
       
    norm_matrix = (b-a).*(feature_mtx - mtx_min)./(mtx_max-mtx_min) + a;
end