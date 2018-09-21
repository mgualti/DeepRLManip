classdef clsLearning
    
    properties
        
        handsList;
        imgList;
        labels;
        objUIDList;
        imageCamPos; % cam pos in local reference frame of hand
        
    end
    
    methods
        
        function lout = clsLearning()
            handList = [];
            imgList = [];
            labels = [];
            objUIDList = [];
            imageCamPos = [];
        end
        
        function l = importFromHands(l,hands)
            handsList = hands.getAllHands();
            labels = hands.getAntipodalLabels(handsList);
            imageList = hands.getImageList(handsList);
            objUIDList = cell(size(labels));
            objUIDList(:) = {hands.objUID};
            imageCamPos = hands.getImageCamPos();
            
            l.handsList = [l.handsList; handsList];
            l.labels = [l.labels labels];
            l.imgList = [l.imgList imageList];
            l.objUIDList = [l.objUIDList objUIDList];
            l.imageCamPos = [l.imageCamPos imageCamPos];
        end
        
        function l = importFromClsLearning(l,limport)
            l.handsList = [l.handsList; limport.handsList];
            l.labels = [l.labels limport.labels];
            l.imgList = [l.imgList limport.imgList];
            l.objUIDList = [l.objUIDList limport.objUIDList];
            l.imageCamPos = [l.imageCamPos limport.imageCamPos];
        end
        

        % Get number of elts in this class
        function out = num(l)
            out = size(l.labels,2);
        end
        
        % Randomly sub sample elts in this class
        function l = subSample(l,numSamples)
            if numSamples > l.num()
                error('clsLearning.subSample: numSamples must be less than number of elts in clsLearning');
            end
            elts2keep = randperm(l.num(),numSamples);
            l = l.prune(elts2keep);
        end
        
        % Prune elts
        % input: idx2keep -> indices of elts NOT to prune.
        function l = prune(l,idx2keep)
            l.handsList = l.handsList(idx2keep,:);
            l.labels = l.labels(idx2keep);
            l.imgList = l.imgList(idx2keep);
            l.objUIDList = l.objUIDList(idx2keep);
            l.imageCamPos = l.imageCamPos(:,idx2keep);
        end
        
        % Get a random train/test split
        function [idxTrain, idxTest] = getSplit(l,fractionTrain)
            if fractionTrain > 1
                error('clsLearning.getSplit: fractionTest must be no greater than one');
            end
            numTrain = floor(l.num() * fractionTrain);
            numTest = l.num() - numTrain;
            idxTest = randperm(l.num(),numTest);
            idxTrain = setdiff(1:l.num(),idxTest);
        end
        
        function predictionList = readFromCAFFE(l,folder)
            predictionList = importdata([folder '/predictionList.txt']);
        end

        % Write this class to the temp directory for use by caffe
        function writeToCAFFE(l, caffeFolder, trainList, testList)
            
            % Remove ./temp directory if it exists
%             temproot = ['~/projects/gpdinc/data/temp'];
            if exist(caffeFolder)
                rmdir(caffeFolder,'s');
            end
            
            l.writeImageList(caffeFolder,'train',trainList);
            l.writeImageList(caffeFolder,'test',testList);
            l.writeImages(caffeFolder,[trainList testList]);
            
        end
                
        % Write images in handList to a specified directory as image files.
        % This is used to get our data into CAFFE.        
        % input: foldername -> folder where images will be stored
        %        handList -> list of images to write
        function writeImages(l, foldername, idx)

            % create jpgs directory as appropriate
            datapath = [foldername '/jpgs'];
            if ~exist(datapath)
                mkdir(datapath);
            end

            for i=1:size(idx,2)
                image = l.imgList{idx(i)};
                imfilename = l.getImgFilename(idx(i));
                im = uint8(image);
                save([datapath '/' imfilename], 'im', '-mat');
            end

        end
        
        % Write a txt file containing a list of filenames to <foldername> directory.
        % input:
        %        foldername -> folder where train/test indices will be
        %                       written
        %        filename -> name of file
        %        idx -> indices of images/labels to be written
        function writeImageList(l, foldername, filename, idx)

            % create jpgs directory as appropriate
            datapath = [foldername '/jpgs'];
            if ~exist(datapath)
                mkdir(datapath);
            end

            txt = [];
            for i=1:size(idx,2)
                imfilename = l.getImgFilename(idx(i));
                txt{i} = [imfilename ' ' num2str(l.labels(idx(i)))];
            end
            fid = fopen([foldername '/' filename '.txt'], 'w');
            fmtString = '%s\n';
            fprintf(fid, fmtString, txt{:});
            fclose(fid);

        end
        
        % Given an index into clsLearning, calculate filename for image
        % file.
        % input:
        %        idx -> index into clsLearning of elt for which to get
        %        filename
        function imfilename = getImgFilename(l,idx)
            objUID = l.objUIDList{idx};
            UID = [objUID '_' int2str(idx)];
            imfilename = ['img_' UID '.mat'];
        end
        
        
        % Get CAFFE predictions.
        function predictions = getPredictions(l, modelFile, weightsFile, gpuId, idx)
            if nargin < 5
                idx = 1:l.num;
            end            
            predictionList = l.getScores(modelFile, weightsFile, gpuId, idx);
            [~,labelsPredicted] = max(predictionList');
            predictions = labelsPredicted - 1;            
        end
        
        % Forward propagate the caffe network and get scores. 
        % input: idx -> indices of images/labels to predict.
        %        modelFile -> Name and path of test.prototxt file.
        %        weightsFile -> Name and path of .caffemodel file.
        %        gpuId -> ID of the GPU to use (usually 0). Set to -1 to
        %                 use CPU.
        % output: predictionMatrix <- 2 x length(idx) matrix of predictions
        function predictionMatrix = getScores(l, modelFile, weightsFile, gpuId, idx)
            
            if nargin < 5
                idx = 1:l.num;
            end
            
            if gpuId < 0
                caffe.set_mode_cpu();
            else
                caffe.set_mode_gpu()
                caffe.set_device(gpuId)
            end
            
            net = caffe.Net(modelFile, weightsFile, 'test');
            dataShape = net.blobs('data').shape;
            batchSize = dataShape(end);
            
            predictionMatrix = zeros(length(idx), 2);
            inputImage = zeros(dataShape);
            j = 1;
            
            for i=1:length(idx)
                image = l.imgList{idx(i)};
                image = permute(image, [2 1 3]);
                inputImage(:,:,:,j) = image;
                
                if j < batchSize && i < length(idx)
                    j = j + 1;
                    continue
                end
                
                probs = net.forward({inputImage});
                predictionMatrix(1+i-j:i, :) = probs{1}(:, 1:j)';
                j = 1;
            end
            
            caffe.reset_all();
        end
        
    end
    
end
