                              Order-Preserving Wasserstein Discriminant Analysis

1. Introduction.

This package includes the prototype MATLAB and Python codes for experiments on the ChaLearn dataset, described in

	"Order-Preserving Wasserstein Discriminant Analysis", Bing Su, Jiahuan Zhou, and Ying Wu. ICCV, 2019.



2.  Usage & Dependency.

- LinearOWDA

  Dependency:
     vlfeat-0.9.18
     libsvm-3.20
     liblinear-1.96
  
  tested under Windows 7 x64, Matlab R2015b.

  Usage:
     1. Download the folder "datamat" from "https://pan.baidu.com/s/1mjkonfeJMojoUGnMNYpXpw" and put it under this folder "LinearOWDA", which contains the organized version of the 100-dimensional frame-wide features provided in "https://bitbucket.org/bfernando/videodarwin" (described in "B. Fernando, E. Gavves, J. O. M., A. Ghodrati, and T. Tuytelaars,¡°Modeling video evolution for action recognition,¡± CVPR, 2015."); 
     2. Add the installation paths of the packages at the beginning of "EvaluateOPWDA.m" below "TODO: add path";
     3. Run the main code in Matlab:
        EvaluateOPWDA


- DeepOWDA

  Dependency:
     vlfeat-0.9.21
     libsvm-3.23
     liblinear-2.21
     Python
     Keras with the Theano backend

  tested under Linux Ubuntu 16.04.2, Matlab R2018a.

  Usage:
     1. Download the folder "datamat" from "https://pan.baidu.com/s/1mjkonfeJMojoUGnMNYpXpw" and put it under this folder "DeepOWDA", which contains the organized version of the 100-dimensional frame-wide features provided in "https://bitbucket.org/bfernando/videodarwin" (described in "B. Fernando, E. Gavves, J. O. M., A. Ghodrati, and T. Tuytelaars,¡°Modeling video evolution for action recognition,¡± CVPR, 2015.");
     2. Add the installation paths of the packages at the beginning of "EvaluateDeepOPWDA_ChaLearn.m" below "TODO: add path"; Modify all absolute paths in (.py and .m) codes to your custom paths;
     3. Run the main code in Matlab:
        EvaluateDeepOPWDA_ChaLearn 



3. License & disclaimer.

    The codes and data can be used for research purposes only. This package is strictly for non-commercial academic use only.



4. Notice

1) We utilized or adapted some toolboxes, data, and codes, such as https://github.com/bobye/WBC_Matlab, https://github.com/gpeyre/2014-SISC-BregmanOT, https://github.com/VahidooX/DeepLDA, https://bitbucket.org/bfernando/videodarwin, which are all publicly available. Please also check the license of them if you want to make use of this package.

2) On a new dataset, if the prompt 'NaN distance!' appears or nan loss occurs, it means that when calculating the OPW distance, some intermediate entries on denominator exceeds the machine-precision
limit. You may need to adjust the parameters of the OPW distance (often reduce the value of lambda_1), and/or normalize, scale, or centralize the input features in sequences.



5. Citations

Please cite the following papers if you use the codes:

Bing Su, Jiahuan Zhou, and Ying Wu, "Order-Preserving Wasserstein Discriminant Analysis," IEEE International Conference on Computer Vision, 2019.