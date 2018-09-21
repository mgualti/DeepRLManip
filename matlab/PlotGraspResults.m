function PlotGraspResults()

    %% Parameters
    
    resultsFileName = '../results.mat';
    saveFilePrefix = '2018-06-11-A';
    saveFilePostfixes = {'AverageReturn', 'TestLosses', 'AverageTestLoss', 'TrainingRunTime', 'Exploration', 'DatabaseSize'};
    figsToSave = [];

    %% Load

    close('all');

    if ~exist(resultsFileName, 'file')
        disp([resultsFileName ' not found.']);
        return
    end
    load(resultsFileName);
    figs = [];

    %% Plot Average Return

    figs = [figs, figure]; hold('on');
    plot(avgReturn, '-x', 'linewidth', 2);
    grid('on');
    xlabel('Training Round');
    title('Average Return');
    
    %% Plot Loss
    
    testLoss = testLoss0;
    if ~isempty(testLoss)
        
        iteration = 1:size(testLoss,2);
        iteration = iteration*100; % depends on python code
        
        figs = [figs, figure]; hold('on');
        title('Test Losses');
        plot(iteration, testLoss');
        grid('on');
        xlabel('Caffe Iteration'); ylabel('Loss');

        figs = [figs, figure]; hold('on');
        title('Average Test Loss');
        plot(mean(testLoss,2), '-x', 'linewidth', 2);
        grid('on'); xlabel('Training Round'); ylabel('Loss');
    end

    %% Plot Run Time and Database 

    figs = [figs, figure]; hold('on');
    title('Training Run Time');
    plot(roundTime, '-x', 'linewidth', 2);
    plot(ones(size(roundTime))*mean(roundTime), '--', 'linewidth', 2);
    grid('on'); xlabel('Training Round'); ylabel('Time (s)');
    
    figs = [figs, figure]; hold('on');
    title('Exploration');
    plot(epsilonRound, '-x', 'linewidth', 2);
    grid('on'); xlabel('Training Round'); ylabel('\epsilon');
    
    figs = [figs, figure]; hold('on');
    title('Database Size');
    plot(databaseSize, '-x', 'linewidth', 2);
    grid('on');
    
    %% Saving Images
    
    for idx=1:length(figsToSave)
        saveas(figs(figsToSave(idx)), ['../../notebook/figures-2/' ...
            saveFilePrefix '-' saveFilePostfixes{figsToSave(idx)} '.png']);
    end
    
    if ~isempty(figsToSave)
        system(['mv ' resultsFileName ' ../results-2/' saveFilePrefix ...
            '-results.mat']);
    end
    
end