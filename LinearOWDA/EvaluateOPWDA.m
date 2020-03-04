charnum = 20;
classnum = charnum;
dim = 100;
rankdim = 58;
CVAL = 1;

% TODO Add paths
addpath('E:/BING/ActionRecognition/FrameWideFeatures/vlfeat-0.9.18/toolbox');
vl_setup();
addpath('E:/BING/ActionRecognition/FrameWideFeatures/liblinear-1.96/matlab');
addpath('E:/BING/ActionRecognition/FrameWideFeatures/libsvm-3.20/matlab');

delta = 1;
lambda1 = 50;
lambda2 = 0.1;
bcnum = 8;

options.init_method='average'; % {'kmeans', 'mvnrnd'}
options.support_size = bcnum;
options.ibp_max_iters = 50;
options.lambda1 = lambda1;
options.lambda2 = lambda2;
options.delta = delta;
options.max_support_size=options.support_size;

load('./datamat/trainset.mat');
load('./datamat/trainsetnum.mat');
load('./datamat/testset.mat');
load('./datamat/testsetnum.mat');

load('./datamat/testsetdata.mat');
load('./datamat/testsetlabel.mat');
load('./datamat/testsetdatanum.mat');

load(['./datamat/traindatamean.mat']);
trainset_m = trainset;
for c=1:classnum
    for m = 1:trainsetnum(c)
        trainset_m{c}{m} = bsxfun(@minus, trainset{c}{m}, traindatamean);
    end
end
testsetdata_m = testsetdata;
for m = 1:testsetdatanum
    testsetdata_m{m} = bsxfun(@minus, testsetdata{m}, traindatamean);
end

W = WassersteinBarycenterDA(trainset_m,bcnum,options);
%W2 = W;
for m = 1:size(W,2)
    W(:,m) = W(:,m)./norm(W(:,m));
end


dnum = size([5:10:rankdim],2);
WSDA_SVM_map = zeros(1,dnum);
WSDA_SVM_acc = zeros(1,dnum);
WSDA_SVM_acc_best = zeros(1,dnum);
WSDA_SVM_Recall_b = zeros(1,dnum);
WSDA_SVM_Precision_b = zeros(1,dnum);
WSDA_SVM_F_b = zeros(1,dnum);

WSDA_NN_map = zeros(1,dnum);
WSDA_NN_acc_1 = zeros(1,dnum);
WSDA_NN_acc = cell(1,dnum);
WSDA_knn_time = zeros(1,dnum);



rankdim = size(W,2);
dcount = 0;
 for downdim = [5:10:rankdim]           
    transMatrix = W(:,[1:downdim]);
    dcount = dcount + 1;
    %% ±ä»»    
    traindownset = cell(1,classnum);
    testdownsetdata = cell(1,testsetdatanum);
    for j = 1:charnum
        traindownset{j} = cell(trainsetnum(j),1);
        for m = 1:trainsetnum(j)
            traindownset{j}{m} = trainset_m{j}{m} * transMatrix;
        end
    end
    for j = 1:testsetdatanum
        testdownsetdata{j} = testsetdata_m{j} * transMatrix;
    end
    %% SVM
    [WSDA_SVM_map(dcount), WSDA_SVM_acc(dcount), WSDA_SVM_acc_best(dcount), WSDA_SVM_Recall_b(dcount), WSDA_SVM_Precision_b(dcount), WSDA_SVM_F_b(dcount)] = RankpoolingClassifier(classnum,downdim,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel);
    
    %% NN
    [WSDA_NN_map(dcount),WSDA_NN_acc{dcount},WSDA_knn_time(dcount)] = NNClassifier(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options);
    WSDA_NN_acc_1(dcount) = WSDA_NN_acc{dcount}(1);
 end

fprintf('When the dimension is reduced to 5, 15, 25, 35, 45, 55, respectively, by Linear OWDA:\n');
fprintf('Classification using SVM classifier:\n');
fprintf('Accuracy is '); 
fprintf('%.4f ',WSDA_SVM_acc_best);
fprintf('\n');
fprintf('MAP is '); 
fprintf('%.4f ',WSDA_SVM_map);
fprintf('\n');
fprintf('Precision is '); 
fprintf('%.4f ',WSDA_SVM_Precision_b);
fprintf('\n');
fprintf('Recall is '); 
fprintf('%.4f ',WSDA_SVM_Recall_b);
fprintf('\n');
fprintf('F-score is '); 
fprintf('%.4f ',WSDA_SVM_F_b);
fprintf('\n');

fprintf('Classification using 1 nearest neighbor classifier with OPW distance:\n');
fprintf('Accuracy is '); 
fprintf('%.4f ',WSDA_NN_acc_1);
fprintf('\n');
fprintf('MAP is '); 
fprintf('%.4f ',WSDA_NN_map);
fprintf('\n');