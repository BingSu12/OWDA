charnum = 20;
classnum = charnum;
dim = 100;
rankdim = 58;
CVAL = 1;
downdimpool = [5:10:rankdim-1];


delta = 1;
lambda1 = 50;
lambda2 = 0.1;
options.max_iters = 50;
options.err_limit = 10^(-6);
options.lambda1 = lambda1;
options.lambda2 = lambda2;
options.delta = delta;

bcnum = 8;
options.init_method='average'; % {'kmeans', 'mvnrnd'}
options.support_size = bcnum;
options.ibp_max_iters = 50;
options.max_support_size=options.support_size;


% TODO: add path
addpath('/home/sub/vlfeat-0.9.21/toolbox');
vl_setup();
addpath('/home/sub/libsvm-3.23/matlab');
addpath('/home/sub/liblinear-2.21/matlab');


root = '/data/Bing/DeepLDA-master/code/datamat/ChaLearn';

load([root '/trainset.mat']);
load([root '/trainsetnum.mat']);
load([root '/testset.mat']);
load([root '/testsetnum.mat']);
load([root '/testsetdata.mat']);
load([root '/testsetlabel.mat']);
load([root '/testsetdatanum.mat']);

% load('./datamat/train_data_mean.mat');
% trainset_m = trainset;
% for c=1:classnum
%     for m = 1:trainsetnum(c)
%         trainset_m{c}{m} = bsxfun(@minus, trainset{c}{m}, traindatamean);
%     end
% end
% testsetdata_m = testsetdata;
% for m = 1:testsetdatanum
%     testsetdata_m{m} = bsxfun(@minus, testsetdata{m}, traindatamean);
% end

trainset_m = trainset;
testsetdata_m = testsetdata;
for c = 1:classnum
    for i = 1:trainsetnum(c)
        trainset_m{c}{i} = trainset{c}{i}/10;
    end
end
for i = 1:testsetdatanum
    testsetdata_m{i} = testsetdata{i}/10;
end




% W = WassersteinBarycenterDA(trainset_m,bcnum,options);
% save('./datamat/WSDAtrans_l12_n.mat','W');

[traindatafull, trainlabelfull, Yinter] = OWDAAlignFea(trainset_m,bcnum,options);
save([root '/traindatafull.mat'],'traindatafull','trainlabelfull','Yinter');

disp('Warming Up Done!');

testdatafull = [];
%testlabels = [];
%testsetdatanum = length(testsetdata_m);
for i = 1:testsetdatanum
    testdatafull = [testdatafull; testsetdata_m{i}];
    %testlabels = [testlabels; zeros(size(testsetdata{i},1),1)];
end
save([root '/testdatafull.mat'],'testdatafull');

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



%rankdim = size(W,2);
dcount = 0;
for downdim = [5:10:rankdim]
    dcount = dcount + 1;
    system('echo $LD_LIBRARY_PATH')
    system(['unset LD_LIBRARY_PATH; unset MKL_NUM_THREADS; unset MKL_DOMAIN_NUM_THREADS; python DeepOWDA_ChaLearn.py --root ' root ' --out_dim ' num2str(downdim) ' --n_components ' num2str(downdim)])
    %system(['unset LD_LIBRARY_PATH; unset MKL_NUM_THREADS; unset MKL_DOMAIN_NUM_THREADS; python DeepOWDA.py --root ' root ' --out_dim ' num2str(downdim) ' --n_components ' num2str(downdim)])
    
    load([root '/TransFeatures.mat']);
    [traindownset,testdownsetdata] = Pre_deepdata_Reverse_full_aus(classnum, train_feature, test_feature, trainsetnum, trainset_m, testsetdatanum, testsetdata_m);


    %% SVM
    [WSDA_SVM_map(dcount), WSDA_SVM_acc(dcount), WSDA_SVM_acc_best(dcount), WSDA_SVM_Recall_b(dcount), WSDA_SVM_Precision_b(dcount), WSDA_SVM_F_b(dcount)] = RankpoolingClassifier(classnum,downdim,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel);
    
    %% NN
    [WSDA_NN_map(dcount),WSDA_NN_acc{dcount},WSDA_knn_time(dcount)] = NNClassifier(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options);
    WSDA_NN_acc_1(dcount) = WSDA_NN_acc{dcount}(1);
 end
save('deepOWDA_ChaLearn_Norm2_ite500_batch500.mat','WSDA_SVM_map','WSDA_SVM_acc','WSDA_SVM_acc_best','WSDA_SVM_Recall_b','WSDA_SVM_Precision_b','WSDA_SVM_F_b','WSDA_NN_map','WSDA_NN_acc','WSDA_NN_acc_1','WSDA_knn_time');

%% NM classifier
%[Map,Acc,knn_averagetime] = NNClassifier(classnum,trainset,trainsetnum,testsetdata,testsetdatanum,testsetlabel,options);

%% SVM classifier
%[Map, Acc, Acc_best, Recall_b, Precision_b, F_b] = RankpoolingClassifier(classnum,dim,trainset,trainsetnum,testsetdata,testsetdatanum,testsetlabel)
