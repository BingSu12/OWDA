function [Map, Acc, Acc_best, Recall_b, Precision_b, F_b] = RankpoolingClassifier(classnum,dim,trainset,trainsetnum,testsetdata,testsetdatanum,testsetlabel)
    CVAL = 1;
    testsetlabelori = testsetlabel;
    testsetlabel = getLabel(testsetlabelori);
    
    ALL_Data_mbh = cell(1,classnum);
    for c = 1:classnum
        temp_all_data_mbh = zeros(trainsetnum(c),2*5*dim);
        for sample_count = 1:trainsetnum(c)
            W = genRepresentation(trainset{c}{sample_count},CVAL);
            temp_all_data_mbh(sample_count,:) = W';
        end
        ALL_Data_mbh{c} = temp_all_data_mbh;
    end

    ALL_Data_mbh_test = zeros(testsetdatanum,2*5*dim);
    for sample_count = 1:testsetdatanum
        W = genRepresentation(testsetdata{sample_count},CVAL);
        ALL_Data_mbh_test(sample_count,:) = W';
    end

    ALL_Data_full_mbh = [];
    classlabel = [];
    for c=1:classnum
        classlabel = [classlabel; zeros(trainsetnum(c),1)+c];
        ALL_Data_full_mbh = [ALL_Data_full_mbh; ALL_Data_mbh{c}];       
        if size(ALL_Data_mbh{c},1)~=trainsetnum(c)
            disp('Some training data missing!');
        end
    end    
    classlabel = [classlabel; testsetlabelori];
    ALL_Data_full_mbh = [ALL_Data_full_mbh; ALL_Data_mbh_test]; 

    trn = zeros(numel(classlabel),1);
    tst = zeros(numel(classlabel),1);
    trn([1:sum(trainsetnum)],1) = 1;
    tst([sum(trainsetnum)+1:end],1) = 1;

    ALL_Data_cell = cell(1,1);
    ALL_Data_cell{1} = ALL_Data_full_mbh;

    classid = getLabel(classlabel);      
    %[trn,tst] = generateTrainTest(classlabel);
    trn_indx  = find(trn>0);
    test_indx = find(tst>0);
    trainLabels = classlabel(trn_indx,:);
    testLabels = classlabel(test_indx,:);
    Options.KERN = 0;    % non linear kernel
    Options.Norm =2;   % L2 normalization

    if Options.KERN == 5        
        for ch = 1 : size(ALL_Data_cell,2)    
            x = ALL_Data_cell{ch};            
            ALL_Data_cell{ch} = sqrt(abs(x));
        end
    end  	

    if Options.Norm == 2       
         for ch = 1 : size(ALL_Data_cell,2)                 
            ALL_Data_cell{ch} = normalizeL2(ALL_Data_cell{ch});
        end
    end  

    if Options.Norm == 1       
         for ch = 1 : size(ALL_Data_cell,2)                 
            ALL_Data_cell{ch} = normalizeL1(ALL_Data_cell{ch});
        end
    end 

    % if there are multiple features (mbh,hog,hof,trj) then add weights to them
    if size(ALL_Data_cell,2) == 1
        weights = 1;
    end

    if size(ALL_Data_cell,2) == 2 || size(ALL_Data_cell,2) == 6 
        weights = [0.5 0.5];
    end

    if size(ALL_Data_cell,2) > 2 && size(ALL_Data_cell,2) ~= 6
        nch = size(ALL_Data_cell,2) ;
        weights = ones(1,nch) * 1/nch;
    end      

    TrainClass = classid(trn_indx,:);
    TestClass = classid(test_indx,:);      
    for ch = 1 : size(ALL_Data_cell,2)        
        ALL_Data = ALL_Data_cell{ch};
        TrainData = ALL_Data(trn_indx,:);        
        TestData = ALL_Data(test_indx,:);

        TrainData_Kern_cell{ch} = [TrainData * TrainData'];   %chi_square_kernel(TrainData);  
        TestData_Kern_cell{ch} = [TestData * TrainData'];    %chi_square_kernel_2(TestData,TrainData);                    
        clear TrainData; clear TestData; clear ALL_Data;            
    end

    TrainData_Kern = zeros(size(TrainData_Kern_cell{1}));
    TestData_Kern = zeros(size(TestData_Kern_cell{1}));
     for ch = 1 : size(ALL_Data_cell,2)     
         TrainData_Kern = TrainData_Kern + weights(1,ch) * TrainData_Kern_cell{ch};
         TestData_Kern = TestData_Kern + weights(1,ch) * TestData_Kern_cell{ch};
     end    
    nTrain = 1 : size(TrainData_Kern,1);
    TrainData_Kern = [nTrain' TrainData_Kern];         
    nTest = 1 : size(TestData_Kern,1);
    TestData_Kern = [nTest' TestData_Kern];

    for cl = 1 : size(classid,2)            
        trnLBLB = TrainClass(:,cl);
        testLBL = TestClass(:,cl);        
        %for wi = 1 : size(weights,1)
            C = 100;
            ap_class(cl) = train_and_classify2(TrainData_Kern,TestData_Kern,trnLBLB,testLBL,C);       
        %end                   
    end
    for cl = 1 : size(classid,2)
            %fprintf('%s = %1.2f \n',actionName{cl},ap_class(cl));
            fprintf('%d = %1.4f \n',cl,ap_class(cl));
    end
    fprintf('mean = %1.4f \n',mean(ap_class));
    Map = mean(ap_class);

    model = svmtrain(trainLabels, TrainData_Kern, sprintf('-t 4 -c %1.6f  -q ',C));
    [~, acc, scores] = svmpredict(testLabels, TestData_Kern ,model);
    Acc = acc(1); 

    TrainClass_ALL = classid(trn_indx,:);
    TestClass_ALL = classid(test_indx,:);   
    [~,TrainClass] = max(TrainClass_ALL,[],2);
    [~,TestClass] = max(TestClass_ALL,[],2);   		

    for ch = 1 : size(ALL_Data_cell,2)        
        ALL_Data = ALL_Data_cell{ch};
        TrainData = ALL_Data(trn_indx,:);        
        TestData = ALL_Data(test_indx,:);

        TrainData_Kern_cell{ch} = [TrainData * TrainData'];    
        TestData_Kern_cell{ch} = [TestData * TrainData'];                        
        clear TrainData; clear TestData; clear ALL_Data;            
    end

    for wi = 1 : size(weights,1)
        TrainData_Kern = zeros(size(TrainData_Kern_cell{1}));
        TestData_Kern = zeros(size(TestData_Kern_cell{1}));
            for ch = 1 : size(ALL_Data_cell,2)     
                TrainData_Kern = TrainData_Kern + weights(wi,ch) * TrainData_Kern_cell{ch};
                TestData_Kern = TestData_Kern + weights(wi,ch) * TestData_Kern_cell{ch};
            end
            [precision(wi,:),recall(wi,:),acc(wi) ] = train_and_classify3(TrainData_Kern,TestData_Kern,TrainClass,TestClass,classnum);       
    end          

    [~,indx] = max(acc);
    indx = 1;
    precision = precision(indx,:);
    recall = recall(indx,:); 
    F = 2*(precision .* recall)./(precision+recall);
    F(find(isnan(F)))=0;
    fprintf('Mean F score = %1.4f\n',mean(F));
    %save(sprintf('results.mat'),'precision','recall','F');
    Acc_best = max(acc);
    Recall_b = mean(recall);
    Precision_b = mean(precision);
    F_b = mean(F);
end


function [precision,recall] = thisclass_pre_recall(label,predicted)
    tru_pos = sum((predicted>=0).*(label==1));
    false_pos = sum((predicted>=0).*(label~=1));
    false_neg = sum((predicted<0).*(label==1));
    precision = tru_pos/(tru_pos+false_pos);
    recall = tru_pos/(tru_pos+false_neg);
end

% function W = genRepresentation(data,CVAL)
%     OneToN = [1:size(data,1)]';    
%     Data = cumsum(data);
%     Data = Data ./ repmat(OneToN,1,size(Data,2));
%     W_fow = liblinearsvr(getNonLinearity(Data),CVAL,2); 			
%     order = 1:size(data,1);
%     [~,order] = sort(order,'descend');
%     data = data(order,:);
%     Data = cumsum(data);
%     Data = Data ./ repmat(OneToN,1,size(Data,2));
%     W_rev = liblinearsvr(getNonLinearity(Data),CVAL,2); 			              
%     W = [W_fow ; W_rev];
% end

function W = genRepresentation(data,CVAL)
    Data =  zeros(size(data,1)-1,size(data,2));
    for j = 2 : size(data,1)                
        Data(j-1,:) = mean(data(1:j,:));
    end
    Data = vl_homkermap(Data',2,'kchi2');
    Data = Data';

    W_fow = liblinearsvr(Data,CVAL,2); 			
    order = 1:size(data,1);
    [~,order] = sort(order,'descend');
    data = data(order,:);
    Data =  zeros(size(data,1)-1,size(data,2));
    for j = 2 : size(data,1)                
        Data(j-1,:) = mean(data(1:j,:));
    end            
    Data = vl_homkermap(Data',2,'kchi2');
    Data = Data';            
    W_rev = liblinearsvr(Data,CVAL,2); 			              
    W = [W_fow ; W_rev];  

end

function Data = getNonLinearity(Data)
    %Data = sign(Data).*sqrt(abs(Data));
    %Data = vl_homkermap(Data',2,'kchi2');
    %Data =  sqrt(abs(Data));	                	
    Data =  sqrt(Data);	      
end

function X = normalizeL2(X)
	for i = 1 : size(X,1)
		if norm(X(i,:)) ~= 0
			X(i,:) = X(i,:) ./ norm(X(i,:));
		end
    end	   
end

function X = normalizeL1(X)
	for i = 1 : size(X,1)
		if norm(X(i,:),1) ~= 0
			X(i,:) = X(i,:) ./ norm(X(i,:),1);
		end
    end	   
end

function X = rootKernelMap(X)
    X = sqrt(X);
end

function [ap ] = train_and_classify(TrainData_Kern,TestData_Kern,TrainClass,TestClass)
         nTrain = 1 : size(TrainData_Kern,1);
         TrainData_Kern = [nTrain' TrainData_Kern];         
         nTest = 1 : size(TestData_Kern,1);
         TestData_Kern = [nTest' TestData_Kern];
         C = [0.01 0.1 1 5 10 50 100 500 1000];
		 % TODO : Note that here it is best to do the cross validation on training set.
		 %warning('It is best to do the cross validation on training set. Skipping cross validation!');
		 C = [100];
         model = svmtrain(TrainClass, TrainData_Kern, sprintf('-t 4 -c %1.6f  -q ',C));
         [~, acc, scores] = svmpredict(TestClass, TestData_Kern ,model);	                 
         [rc, pr, info] = vl_pr(TestClass, scores(:,1)) ; 
         ap = info.ap;      
end

function [ap ] = train_and_classify2(TrainData_Kern,TestData_Kern,TrainClass,TestClass,C)
         
         %C = [0.01 0.1 1 5 10 50 100 500 1000];
		 % TODO : Note that here it is best to do the cross validation on training set.
		 %warning('It is best to do the cross validation on training set. Skipping cross validation!');
		 %C = [10];
         model = svmtrain(TrainClass, TrainData_Kern, sprintf('-t 4 -c %1.6f  -q ',C));
         [~, acc, scores] = svmpredict(TestClass, TestData_Kern ,model);	                 
         [rc, pr, info] = vl_pr(TestClass, scores(:,1)) ; 
         ap = info.ap;        
end

function [trn,tst] = generateTrainTest(classid)
    trn = zeros(numel(classid),1);
    tst = zeros(numel(classid),1);
    maxC = max(classid);
    for c = 1 : maxC
        indx = find(classid == c);
        n = numel(indx);
        tindx = indx(1:4);
        testindx = indx(5:end);
        trn(tindx,1) = 1;
        tst(testindx,1) = 1;
    end
end

function [X] = getLabel(classid)
    X = zeros(numel(classid),max(classid))-1;
    for i = 1 : max(classid)
        indx = find(classid == i);
        X(indx,i) = 1;
    end
end

function w = liblinearsvr(Data,C,normD)
    if normD == 2
        Data = normalizeL2(Data);
    end
    
    if normD == 1
        Data = normalizeL1(Data);
    end
    % in case it is complex, takes only the real part.	
    N = size(Data,1);
    Labels = [1:N]';
    model = train(double(Labels), sparse(double(Data)),sprintf('-c %d -s 11 -q',C) );
    w = model.w';
end


function [precision,recall,acc ] = train_and_classify3(TrainData_Kern,TestData_Kern,TrainClass,TestClass,classnum)
         nTrain = 1 : size(TrainData_Kern,1);
         TrainData_Kern = [nTrain' TrainData_Kern];         
         nTest = 1 : size(TestData_Kern,1);
         TestData_Kern = [nTest' TestData_Kern];         
         C = [1 10 100 500 1000 ];
         for ci = 1 : numel(C)
             model(ci) = svmtrain(TrainClass, TrainData_Kern, sprintf('-t 4 -c %1.6f -v 2 -q ',C(ci)));               
         end        
         
         [~,max_indx]=max(model);
         
         C = C(max_indx);
         
        
         for ci = 1 : numel(C)
             model = svmtrain(TrainClass, TrainData_Kern, sprintf('-t 4 -c %1.6f  -q ',C(ci)));
             [predicted, acc, scores{ci}] = svmpredict(TestClass, TestData_Kern ,model);	                 
             [precision(ci,:) , recall(ci,:)] = perclass_precision_recall(TestClass,predicted,classnum);
             accuracy(ci) = acc(1,1);
         end        
         
        [acc,cindx] = max(accuracy);   
        scores = scores{cindx};
        precision = precision(cindx,:);
        recall = recall(cindx,:);
end

function [precision , recall] = perclass_precision_recall(label,predicted,classnum)      
    for cl = 1 : classnum
        true_pos = sum((predicted == cl) .* (label == cl));
        false_pos = sum((predicted == cl) .* (label ~= cl));
        false_neg = sum((predicted ~= cl) .* (label == cl));
        precision(cl) = true_pos / (true_pos + false_pos);
        recall(cl) = true_pos / (true_pos + false_neg);
        
    end
end