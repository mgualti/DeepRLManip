% Run this file after running createImages.m
% 
% Given the *_LabeledGraspCandidates.mat and *_Images.mat files created by
% createImages.m, this file writes the data to a directory from which a
% caffe model can be trained.
% 
function trainCaffeMugs

    close all;
    
    categoryroot = './data/MATfiles/';
    folderset = {'mug'};

    % Import data from *_Images.mat files into clsLearning::l
    useParfor = 0;
    tc = clsTrainCaffe(categoryroot,folderset,useParfor,'');
    l = clsLearning();
    l = tc.loadClsLearning(l,10000); % max num examples per object
    
    % Get train/test split on data in clsLearning::l. Write it to the
    % directory.
    [idxTrain, idxTest] = l.getSplit(0.75);
    l.writeToCAFFE('./data/CAFFEfiles',idxTrain,idxTest);
    
% Now, you need to run the following python code from the gpdinc directory:
%
% python generate_lmdbs.py CAFFEfiles 0 12
% python generate_nets.py CAFFEfiles
% python train_lenet.py CAFFEfiles
%
% These python files are a mess, but they should work. I'll try to fix them
% soon.
% 
% At this point, you should be able to look at solver_results.txt and
% view accuracy on the test set as a function of training iteration.
% 
    
end

