function [traindatafull, labeldatafull, Yinter] = OWDAAlignFea(trainsequences,bcnum,options)
    
    %classnum = size(trainsequences,2);
    dim = size(trainsequences{1}{1},2);
    classnum = length(trainsequences);
    %bcnum = 8;
    
    if nargin<3 || isempty(options)
        delta = 1;
        lambda1 = 50;
        lambda2 = 0.1;

        options.init_method='average'; % {'kmeans', 'mvnrnd'}
        options.support_size = bcnum;
        options.ibp_max_iters = 50;
        options.lambda1 = lambda1;
        options.lambda2 = lambda2;
        options.delta = delta;
        options.max_support_size=options.support_size; 
    end
    
    traindatafull = [];
    labeldatafull = [];
    
    %% obtain within-class scatter
    perCnum = zeros(1,classnum);
    ct_barycenter = cell(1,classnum);
    %covariance = cell(1,classnum);
    totalnum = 0;
    Sw = zeros(dim,dim);
    for c = 1:classnum
        [ct,T] = OPW_Barycenter(trainsequences{c},bcnum,options);
        ct_barycenter{c} = ct;
        seqnum = length(trainsequences{c})
        perCnum(c) = seqnum;
        %totalnum = totalnum + seqnum;
        %covariance{c} = zeros(dim,dim);
        for t = 1:seqnum
            seqlen = size(trainsequences{c}{t},1);
            temp_label = [T{t}' zeros(seqlen,1)+c-1];
            traindatafull = [traindatafull; trainsequences{c}{t}];
            labeldatafull = [labeldatafull; temp_label];
            
            %seq_w = ones(seq_length,1)./seq_length;
            %[dist, T] = OPW_w(trainsequences{c}{t},ct.supp',seq_w,ct.w',lambda1,lambda2,delta);
%             for i = 1:seq_length
%                 for j = 1:size(ct.supp,2)
%                     covariance{c} = covariance{c} + T{t}(j,i)*(trainsequences{c}{t}(i,:)-ct.supp(:,j)')'*(trainsequences{c}{t}(i,:)-ct.supp(:,j)');
%                 end
%             end
        end   
        %covariance{c} = covariance{c};
        %Sw = Sw + covariance{c};
    end
    %Sw = Sw./totalnum;
    
    
    
    %% obtain between-class scatter
    Yinter = zeros(classnum,classnum,options.support_size,options.support_size);
    %Sb = zeros(dim,dim);
    for c = 1:classnum-1
        for c2 = c+1:classnum
            %Stemp = zeros(dim,dim);
            [dist, T] = OPW_w(ct_barycenter{c}.supp',ct_barycenter{c2}.supp',ct_barycenter{c}.w',ct_barycenter{c2}.w',options.lambda1,options.lambda2,options.delta,0);
            Yinter(c,c2,:,:) = T;
%             for i = 1:size(ct_barycenter{c}.supp,2)
%                 for j = 1:size(ct_barycenter{c2}.supp,2)
%                     Stemp = Stemp + T(i,j)*(ct_barycenter{c}.supp(:,i)-ct_barycenter{c2}.supp(:,j))*(ct_barycenter{c}.supp(:,i)-ct_barycenter{c2}.supp(:,j))';
%                 end
%             end
            %Sb = Sb + perCnum(c)*perCnum(c2)*Stemp;
        end
    end
    
    
    
    %% LDA
    %maxdowndim = dim;
    %W = 1;
    %W = getTransSD(Sw,Sb);
end