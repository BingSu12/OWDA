function transMatrix = getTransSD(sigmaw,sigmab)

    %%等价方法计算变换矩阵
    dim = size(sigmaw,1);
    p = orth(sigmaw);
    gama1 = p'*sigmaw*p;
    w1 = p*(gama1^(-0.5));
    s2 = w1'*sigmab*w1;
    p2 = orth(s2);
    gama2 = p2'*s2*p2;
    rankdim = rank(gama2);
%     if rankdim<downdim
%         disp('Too many dims have been perserved!');
%     end
    a = zeros(1,rankdim);
    for i=1:rankdim
        a(i)=-gama2(i,i);
    end
    [d,index]=sort(a);
    q = [];
    for i=1:rankdim
        q = [q p2(:,index(i))];
    end
    transMatrix = w1*q;
    % sumTrans = transMatrix2.^2;
    % sumT = sum(sumTrans,1);
    % sumT = sumT.^0.5;
    % for i = 1:downdim
    %     transMatrix2(:,i) = 10*transMatrix2(:,i)/sumT(i);
    % end
    %transMatrix = 10*transMatrix;