classdef GRAB_ROI_Analysis_v3 < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        ROIANALYSISUIFigure        matlab.ui.Figure
        AdjustContrastButton       matlab.ui.control.Button
        PlayButton                 matlab.ui.control.Button
        FrameSlider                matlab.ui.control.Slider
        FrameSliderLabel           matlab.ui.control.Label
        ClearallButton             matlab.ui.control.Button
        ClearROIButton             matlab.ui.control.Button
        LoadROIButton              matlab.ui.control.Button
        SaveROIButton              matlab.ui.control.Button
        BaselineMethodDropDown     matlab.ui.control.DropDown
        BaselinecorrectionLabel    matlab.ui.control.Label
        DefaultDirLabel            matlab.ui.control.Label
        SetDefaultDirectoryButton  matlab.ui.control.Button
        FramePeriodEditField       matlab.ui.control.NumericEditField
        FrameintervalsLabel        matlab.ui.control.Label
        SaveresultsButton          matlab.ui.control.Button
        AnalyzeButton              matlab.ui.control.Button
        DrawROIButton              matlab.ui.control.Button
        LoadTIFButton              matlab.ui.control.Button
        CompareAxes                matlab.ui.control.UIAxes
        RawAxes                    matlab.ui.control.UIAxes
        TraceAxes                  matlab.ui.control.UIAxes
        ImageAxes                  matlab.ui.control.UIAxes
    end

    
    properties (Access = private)
    imStack double          % (h x w x nFrames) imported frames
    roiMask logical         % roi mask
    roiPolygon %images.roi.Polygon   % handle to interactive ROI
    roiPosition double              % Nx2 polygon vertices
    t double                % time vector
    meanF double            % mean fluorescence trace
    dFF double              % deltaF/F0
    F0 double               % baseline
    framePeriod double = 0.2; % seconds per frame
    defaultDir char = '';
    meanF_raw double      % store raw mean trace
    dFF_raw double        % dF/F0 without correction
     ImageHandle matlab.graphics.primitive.Image
    currentFrame (1,1) double = 1              
    isPlaying logical = false       % for play/stop

    end
    
    methods (Access = private)
        
        function y = slidingPercentile(app, x, p, win)
    % Compute running percentile p over window win for vector x

    n = numel(x);
    y = zeros(size(x));
    halfwin = floor(win/2);

    for i = 1:n
        i1 = max(1, i-halfwin);
        i2 = min(n, i+halfwin);
        y(i) = prctile(x(i1:i2), p);
    end
        end 
       
        
        function z = baseline_whittaker(app, y, lambda, w)
    % y: column vector
    % lambda: smoothness (e.g. 1e3 – 1e6)
    % w: weights (same length as y), default all ones

    y = y(:);
    n = numel(y);

    if nargin < 3 || isempty(w)
        w = ones(n,1);
    else
        w = w(:);
    end

    % Difference matrix for second derivative
    E = speye(n);
    D = diff(E, 2);             % (n-2) x n
    W = spdiags(w, 0, n, n);

    % Solve (W + λ DᵀD) z = W y
    C = W + lambda * (D' * D);
    z = C \ (W * y);
            
        end
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Button pushed function: LoadTIFButton
        function LoadTIFButtonPushed(app, event)
             
            % Ask user for file
    startDir = app.defaultDir;
if isempty(startDir) || ~isfolder(startDir)
    startDir = pwd;
end

[fn, fp] = uigetfile({'*.tif;*.tiff','TIF files'}, 'Select TIF stack', startDir);
    
if isequal(fn,0); return; end
    fname = fullfile(fp,fn);

    % Read basic info
    info = imfinfo(fname);
    nTotal = numel(info);

    % Ask whether to use all frames or subset
    choice = questdlg( ...
        sprintf('File has %d frames. Import all or subset?', nTotal), ...
        'Frame selection', ...
        'All frames','Subset...','Cancel','All frames');

    if isempty(choice) || strcmp(choice,'Cancel')
        return;
    end

    firstFrame = 1;
    lastFrame  = nTotal;

    if strcmp(choice,'Subset...')
        answer = inputdlg({ ...
            'First frame to import (1-based):', ...
            sprintf('Last frame to import (<= %d):', nTotal)}, ...
            'Select frame subset', ...
            1, {num2str(1), num2str(nTotal)});
        if isempty(answer); return; end

        firstFrame = max(1, round(str2double(answer{1})));
        lastFrame  = min(nTotal, round(str2double(answer{2})));

        if isnan(firstFrame) || isnan(lastFrame) || firstFrame>lastFrame
            uialert(app.ROIANALYSISUIFigure,'Invalid frame range.','Error');
            return;
        end
    end

    nFrames = lastFrame - firstFrame + 1;

    % Preallocate and read frames
    h = info(1).Height;
    w = info(1).Width;
    app.imStack = zeros(h, w, nFrames, 'double');

    wb = waitbar(0,'Loading frames...');
    for k = 1:nFrames
        frameIdx = firstFrame + k - 1;
       img = imread(fname, frameIdx);   % uint8 or uint16
       app.imStack(:,:,k) = double(img); % raw intensity values, no normalization
        if mod(k,10)==0 || k==nFrames
            waitbar(k/nFrames, wb);
        end
    end
    close(wb);

    % Update frame period from UI (if present)
    if ~isempty(app.FramePeriodEditField)
        app.framePeriod = app.FramePeriodEditField.Value;
    end

    % Show mean projection for ROI drawing
    meanImg = mean(app.imStack, 3);
    imagesc(app.ImageAxes, meanImg);
    axis(app.ImageAxes, 'image');
    colormap(app.ImageAxes, gray);
    title(app.ImageAxes, sprintf('Mean image (%d frames)', nFrames));

    % Clear old ROI & trace if any
    app.roiMask = [];
    cla(app.TraceAxes);

    % Assume app.imStack is now [h w nFrames]
[~, ~, nFrames] = size(app.imStack);
app.currentFrame = 1;

% Compute mean image for default display
meanImg = mean(app.imStack, 3);

% Create or update image handle on ImageAxes
if isempty(app.ImageHandle) || ~isvalid(app.ImageHandle)
    app.ImageHandle = imagesc(app.ImageAxes, meanImg);
    axis(app.ImageAxes,'image');
    colormap(app.ImageAxes, gray);
else
    app.ImageHandle.CData = meanImg;
end
title(app.ImageAxes, sprintf('Mean image (%d frames)', nFrames));

% Fix CLim once (pick one of these):

% Simple min/max:
% app.ImageAxes.CLim = [min(app.imStack(:)) max(app.imStack(:))];

% Enhanced contrast using percentiles:
low  = prctile(app.imStack(:), 1);
high = prctile(app.imStack(:), 99);

app.ImageAxes.CLim = [low high];
% Configure frame slider
app.FrameSlider.Limits = [1 nFrames];
app.FrameSlider.Value  = 1;
app.FrameSlider.MajorTicks = linspace(1, nFrames, min(11,nFrames));
app.FrameSlider.Enable = 'on';

% Optional: reset play state
app.isPlaying = false;
if isprop(app,'PlayButton') && isvalid(app.PlayButton)
    app.PlayButton.Text = 'Play';
end
        end

        % Button pushed function: DrawROIButton
        function DrawROIButtonPushed(app, event)
             if isempty(app.imStack)
        uialert(app.ROIANALYSISUIFigure,'Load a TIF file first.','No data');
        return;
    end

    % Show mean image again (optional)
    meanImg = mean(app.imStack, 3);
    imagesc(app.ImageAxes, meanImg);
    axis(app.ImageAxes, 'image');
    colormap(app.ImageAxes, gray);
    title(app.ImageAxes, 'Draw ROI (double-click to finish)');

    % Delete old ROI if needed
    if ~isempty(app.roiPolygon) && isvalid(app.roiPolygon)
        delete(app.roiPolygon);
    end

    % Let user draw polygon ROI
    app.roiPolygon = drawpolygon(app.ImageAxes);
    if isempty(app.roiPolygon)
        return;
    end

    % Save ROI info
    app.roiMask     = createMask(app.roiPolygon);
    app.roiPosition = app.roiPolygon.Position;
        end

        % Button pushed function: AnalyzeButton
        function AnalyzeButtonPushed(app, event)
            %-------------------------------
    % Basic checks
    %-------------------------------
    if isempty(app.imStack)
        uialert(app.ROIANALYSISUIFigure,'Load a TIF file first.','No data');
        return;
    end
    if isempty(app.roiMask)
        uialert(app.ROIANALYSISUIFigure,'Draw an ROI first.','No ROI');
        return;
    end

    [h, w, nFrames] = size(app.imStack);
    mask = app.roiMask;
    if ~isequal(size(mask), [h, w])
        uialert(app.ROIANALYSISUIFigure,'ROI mask size does not match image size. Redraw ROI.','ROI error');
        return;
    end

    maskLinear = mask(:);
    nPix = sum(maskLinear);
    if nPix == 0
        uialert(app.ROIANALYSISUIFigure,'ROI has zero area.','ROI error');
        return;
    end

    %-------------------------------
    % Mean intensity (RAW)
    %-------------------------------
    meanF_local = zeros(nFrames,1);
    for k = 1:nFrames
        frame = app.imStack(:,:,k);
        meanF_local(k) = sum(frame(:).*maskLinear) / nPix;
    end

    app.meanF_raw = meanF_local;
    app.t = (0:nFrames-1)' * app.framePeriod;

    % Baseline from first 50 frames (for "raw" dF/F0)
    nBase = min(50, nFrames);
    F0_raw = mean(meanF_local(1:nBase));
    app.dFF_raw = (meanF_local - F0_raw) / F0_raw;

    %-------------------------------
    % Choose baseline correction method
    %-------------------------------
    method = app.BaselineMethodDropDown.Value;

    meanF_corr = meanF_local;                   % default: unchanged
    baselineVec = F0_raw * ones(size(meanF_local));  % default for plots

    switch method

        case 'None'
            % nothing to do, already set

        %case 'Running baseline'
            %winSeconds = 20;  % tweak as needed
            %winFrames  = max(5, round(winSeconds / app.framePeriod));
            %baselineVec = movmedian(meanF_local, winFrames);
            % dF/F will be computed vs this baseline later
            % meanF_corr stays equal to meanF_local

        case 'Linear detrend'
            p      = polyfit(app.t, meanF_local, 1);
            trend  = polyval(p, app.t);
            meanF_corr  = meanF_local ./ trend * trend(1);   % normalize
            baselineVec = trend;

        case 'Exponential bleach'
            % Fit F(t) = a*exp(-t/b) + c to baseline only
            t = app.t;

            % 1) Find peak = main event
            [~, peakIdx] = max(meanF_local);
            peakTime = t(peakIdx);

            % 2) Exclude ±excludeWindow sec around peak from fit
            excludeWindow = 8;   % seconds, tweak as needed
            excludeMask = (t > (peakTime - excludeWindow)) & ...
                          (t < (peakTime + excludeWindow));
            baselineMask = ~excludeMask;

            % 3) Define exponential model
            ft = fittype('a*exp(-x/b) + c', ...
                         'independent','x', ...
                         'dependent','y');

            tBase = t(baselineMask);
            FBase = meanF_local(baselineMask);

            a0 = FBase(1) - FBase(end);
            b0 = (max(tBase) - min(tBase)) / 2;
            c0 = FBase(end);

            fitobj = fit(tBase, FBase, ft, 'StartPoint',[a0 b0 c0]);

            % 4) Evaluate fitted bleach curve over all time points
            baselineVec = feval(fitobj, t);

            % 5) Divide out exponential bleach & renormalize
            meanF_corr = meanF_local ./ baselineVec * baselineVec(1);

        %case 'Running percentile baseline'
    %t = app.t;
    %winSeconds = 16;
    %winFrames  = max(5, round(winSeconds / app.framePeriod));
    %p = 2;  % 10th percentile

    %baselineVec = app.slidingPercentile(meanF_local, p, winFrames);
    %baselineVec(baselineVec <= 0) = eps;

    %meanF_corr = meanF_local ./ baselineVec * baselineVec(1);
    

 case 'Whittaker baseline(AsLS)'
   
    y = meanF_local(:);
    n = numel(y);

    % Smoothing strength (tune 1e2 – 1e5)
    lambda = 1e3;

    % Asymmetry parameter (smaller = more peak suppression)
    p = 0.001;

    % Initialize weights
    w = ones(n,1);

    % Iterate to suppress peaks & isolate smooth baseline
    for iter = 1:10

        % Fit symmetric Whittaker using current weights
        z = app.baseline_whittaker(y, lambda, w);

        % Update weights asymmetrically:
        % points ABOVE baseline (signal peaks) --> small weight
        % points BELOW baseline (bleaching)    --> large weight
        w = p * (y > z) + (1 - p) * (y <= z);
    end

    baselineVec = z;
    baselineVec(baselineVec <= 0) = eps;

    meanF_corr = y ./ baselineVec * baselineVec(1);
    end  
    
    %-------------------------------
    % dF/F0 using corrected trace
    %-------------------------------
    F0_corr = mean(meanF_corr(1:nBase));
    dFF_corr = (meanF_corr - F0_corr) / F0_corr;

    % Store in app properties
    app.meanF = meanF_corr;
    app.F0    = F0_corr;
    app.dFF   = dFF_corr;

    %-------------------------------
    % PLOTTING
    %-------------------------------

    %% 1) Top-right: corrected trace
    cla(app.TraceAxes, 'reset');

    yyaxis(app.TraceAxes,'left');
    plot(app.TraceAxes, app.t, meanF_corr, 'o-', 'DisplayName','Mean F (corrected)');
    ylabel(app.TraceAxes,'Mean ROI intensity (raw counts)');
    hold(app.TraceAxes,'on');
    yline(app.TraceAxes, F0_corr, '--', 'DisplayName','F_0 (corr)');
    hold(app.TraceAxes,'off');

    yyaxis(app.TraceAxes,'right');
    plot(app.TraceAxes, app.t, dFF_corr, '.', 'DisplayName','dF/F_0 (corrected)');
    yline(app.TraceAxes,0,'--');
    ylabel(app.TraceAxes,'\DeltaF/F_0 (corrected)');

    xlabel(app.TraceAxes,'Time (s)');
    title(app.TraceAxes, sprintf('ROI trace (method: %s, baseline = first %d frames)', ...
        method, nBase));
    legend(app.TraceAxes,'Location','best');


    %% 2) Bottom-left: raw trace + baseline/trend
    if isprop(app,'RawAxes') && ~isempty(app.RawAxes) && isvalid(app.RawAxes)
        cla(app.RawAxes, 'reset');
        hold(app.RawAxes,'on');

        plot(app.RawAxes, app.t, meanF_local, 'b-', 'DisplayName','Mean F (raw)');

        switch method
            case 'Running baseline'
                plot(app.RawAxes, app.t, baselineVec, 'r-', 'DisplayName','Running baseline');

            case 'Linear detrend'
                plot(app.RawAxes, app.t, baselineVec, 'r-', 'DisplayName','Linear trend');

            case 'Exponential bleach'
                plot(app.RawAxes, app.t, baselineVec, 'r-', 'DisplayName','Exp bleach fit');

            case 'None'
                yline(app.RawAxes, F0_raw, 'r--', 'DisplayName','F_0 (raw)');

            case 'Running percentile baseline'
    plot(app.RawAxes, app.t, baselineVec, 'r-', 'DisplayName','Running percentile baseline');
        end

        hold(app.RawAxes,'off');
        xlabel(app.RawAxes,'Time (s)');
        ylabel(app.RawAxes,'Mean ROI intensity (raw counts)');
        title(app.RawAxes,'Raw trace & baseline/trend');
        legend(app.RawAxes,'Location','best');
    end


    %% 3) Bottom-right: raw vs corrected dF/F0
    if isprop(app,'CompareAxes') && ~isempty(app.CompareAxes) && isvalid(app.CompareAxes)
        cla(app.CompareAxes, 'reset');
        hold(app.CompareAxes,'on');

        plot(app.CompareAxes, app.t, app.dFF_raw, 'k-', 'DisplayName','dF/F_0 (raw, first 50 frames)');
        plot(app.CompareAxes, app.t, dFF_corr, 'r-', 'DisplayName','dF/F_0 (corrected)');
        yline(app.CompareAxes,0,'--');

        hold(app.CompareAxes,'off');
        xlabel(app.CompareAxes,'Time (s)');
        ylabel(app.CompareAxes,'\DeltaF/F_0');
        title(app.CompareAxes,'Comparison: raw vs corrected');
        legend(app.CompareAxes,'Location','best');
    end


        end

        % Button pushed function: SaveresultsButton
        function SaveresultsButtonPushed(app, event)
            if isempty(app.meanF) || isempty(app.dFF)
        uialert(app.ROIANALYSISUIFigure,'Run the analysis first.','No results');
        return;
    end

    startDir = app.defaultDir;
if isempty(startDir) || ~isfolder(startDir)
    startDir = pwd;
end

[fn, fp] = uiputfile({'*.mat','MAT-file (*.mat)'}, 'Save analysis', startDir);
    if isequal(fn,0); return; end
    savePath = fullfile(fp, fn);

    t = app.t;
    meanF = app.meanF;
    dFF = app.dFF;
    F0 = app.F0;
    roiMask = app.roiMask;

    save(savePath, 't','meanF','dFF','F0','roiMask');

    % Also save CSV next to MAT
    [~, base, ~] = fileparts(savePath);
    csvPath = fullfile(fp, [base '.csv']);
    T = table(t, meanF, dFF, ...
        'VariableNames', {'Time_s','MeanF','dFoverF0'});
    writetable(T, csvPath);

    uialert(app.ROIANALYSISUIFigure, ...
        sprintf('Saved MAT and CSV:\n%s\n%s', savePath, csvPath), ...
        'Saved');
        end

        % Button pushed function: SetDefaultDirectoryButton
        function SetDefaultDirectoryButtonPushed(app, event)
            startDir = app.defaultDir;
    if isempty(startDir) || ~isfolder(startDir)
        startDir = pwd;
    end

    d = uigetdir(startDir, 'Select default working directory');
    if d == 0
        return; % user cancelled
    end

    app.defaultDir = d;
    app.DefaultDirLabel.Text = ['Default directory: ' d];
        end

        % Button pushed function: SaveROIButton
        function SaveROIButtonPushed(app, event)
            if isempty(app.roiMask)
        uialert(app.ROIANALYSISUIFigure, 'Draw or load an ROI first.', 'No ROI');
        return;
    end

    % Choose save location
    startDir = app.defaultDir;
    if isempty(startDir) || ~isfolder(startDir)
        startDir = pwd;
    end

    [fn, fp] = uiputfile('*.mat', 'Save ROI', startDir);
    if isequal(fn,0)
        return;
    end

    % Prepare data
    roiMask     = app.roiMask;
    roiPosition = app.roiPosition;
    imageSize   = size(app.imStack(:,:,1));

    % Save
    save(fullfile(fp, fn), 'roiMask', 'roiPosition', 'imageSize');
        end

        % Button pushed function: LoadROIButton
        function LoadROIButtonPushed(app, event)
            if isempty(app.imStack)
        uialert(app.ROIANALYSISUIFigure, 'Load a TIF file first.', 'No data');
        return;
    end

    % Choose file
    startDir = app.defaultDir;
    if isempty(startDir) || ~isfolder(startDir)
        startDir = pwd;
    end

    [fn, fp] = uigetfile('*.mat', 'Load ROI', startDir);
    if isequal(fn,0)
        return;
    end

    % Load
    S = load(fullfile(fp, fn));

    % Validate contents
    if ~isfield(S, 'roiMask') || ~isfield(S, 'roiPosition') || ~isfield(S, 'imageSize')
        uialert(app.ROIANALYSISUIFigure, 'Selected file does not contain ROI data.', 'Load Error');
        return;
    end

    % Check size compatibility
    thisSize = size(app.imStack(:,:,1));
    if ~isequal(thisSize, S.imageSize)
        uialert(app.ROIANALYSISUIFigure, 'ROI does not match image dimensions.', 'Size Mismatch');
        return;
    end

    % Store in app
    app.roiMask     = S.roiMask;
    app.roiPosition = S.roiPosition;

    % Remove old ROI if present
    if ~isempty(app.roiPolygon) && isvalid(app.roiPolygon)
        delete(app.roiPolygon);
    end

    % Draw loaded ROI on axes
    app.roiPolygon = drawpolygon(app.ImageAxes, 'Position', app.roiPosition);
        end

        % Button pushed function: ClearROIButton
        function ClearROIButtonPushed(app, event)
             % Delete polygon ROI if it exists
    if ~isempty(app.roiPolygon) && isvalid(app.roiPolygon)
        delete(app.roiPolygon);
    end
    app.roiPolygon = [];
    app.roiMask = [];
    app.roiPosition = [];

    % Clear plots
    cla(app.TraceAxes, 'reset');
    cla(app.RawAxes, 'reset');
    cla(app.CompareAxes, 'reset');

    % Optionally restore the mean image
    if ~isempty(app.imStack)
        meanImg = mean(app.imStack,3);
        imagesc(app.ImageAxes, meanImg);
        axis(app.ImageAxes,'image');
        colormap(app.ImageAxes, gray);
        title(app.ImageAxes, sprintf('Mean image (%d frames)', size(app.imStack,3)));
    end
        end

        % Button pushed function: ClearallButton
        function ClearallButtonPushed(app, event)
            % Clear ROI
    if ~isempty(app.roiPolygon) && isvalid(app.roiPolygon)
        delete(app.roiPolygon);
    end

    app.roiPolygon = [];
    app.roiMask = [];
    app.roiPosition = [];

    % Clear image/movie data
    app.imStack = [];
    app.currentFrame = 1;

    % Clear analysis data
    app.meanF = [];
    app.meanF_raw = [];
    app.dFF = [];
    app.dFF_raw = [];
    app.F0 = [];
    app.t = [];

    % Clear axes
    cla(app.ImageAxes, 'reset');
    cla(app.TraceAxes, 'reset');
    cla(app.RawAxes, 'reset');
    cla(app.CompareAxes, 'reset');

    title(app.ImageAxes, 'No image loaded');

    % Disable frame slider if present
    if isprop(app,'FrameSlider')
        app.FrameSlider.Value = 1;
        app.FrameSlider.Enable = 'off';
    end
        end

        % Value changed function: FrameSlider
        function FrameSliderValueChanged(app, event)
            if isempty(app.imStack)
        return;
    end

    idx = round(app.FrameSlider.Value);
    idx = max(1, min(size(app.imStack,3), idx));  % clamp
    app.currentFrame = idx;

    % Update image to this frame (only update CData)
    frameImg = app.imStack(:,:,idx);

    if isempty(app.ImageHandle) || ~isvalid(app.ImageHandle)
        % Create image ONLY once
        app.ImageHandle = imagesc(app.ImageAxes, frameImg);
        axis(app.ImageAxes,'image');
        colormap(app.ImageAxes, gray);
        % IMPORTANT: DO NOT set CLim here—Load TIF should handle it
    else
        % All future calls update image ONLY by modifying CData
        app.ImageHandle.CData = frameImg;
    end

    title(app.ImageAxes, sprintf('Frame %d / %d (use slider)', ...
        idx, size(app.imStack,3)));
        end

        % Button pushed function: PlayButton
        function PlayButtonPushed(app, event)
             if isempty(app.imStack)
        return;
    end

    % Toggle play state
    app.isPlaying = ~app.isPlaying;

    if app.isPlaying
        app.PlayButton.Text = 'Stop';
    else
        app.PlayButton.Text = 'Play';
        return;
    end

    nFrames = size(app.imStack,3);

    while app.isPlaying
        % Advance frame index
        app.currentFrame = app.currentFrame + 1;
        if app.currentFrame > nFrames
            app.currentFrame = 1;   % loop
        end

        % Update slider position
        app.FrameSlider.Value = app.currentFrame;

        % Directly update image (NO imagesc here)
        frameImg = app.imStack(:,:,app.currentFrame);
        app.ImageHandle.CData = frameImg;

        title(app.ImageAxes, sprintf('Frame %d / %d (playing)', ...
            app.currentFrame, nFrames));

        drawnow;                 % let UI update

        % Control playback speed (approx framePeriod)
        pause(app.framePeriod);  % or a fixed value like 0.05
    end

    % If we exit the loop, reset button text
    app.PlayButton.Text = 'Play';
        end

        % Button pushed function: AdjustContrastButton
        function AdjustContrastButtonPushed(app, event)
            if isempty(app.ImageHandle) || ~isvalid(app.ImageHandle)
        uialert(app.UIFigure,'No image loaded.','Error');
        return;
    end

    % Launch interactive contrast adjustment
    imcontrast(app.ImageAxes);
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create ROIANALYSISUIFigure and hide until all components are created
            app.ROIANALYSISUIFigure = uifigure('Visible', 'off');
            app.ROIANALYSISUIFigure.Color = [0.9412 0.9412 0.9412];
            app.ROIANALYSISUIFigure.Position = [100 100 1450 780];
            app.ROIANALYSISUIFigure.Name = 'ROI ANALYSIS ';
            app.ROIANALYSISUIFigure.Theme = 'light';

            % Create ImageAxes
            app.ImageAxes = uiaxes(app.ROIANALYSISUIFigure);
            title(app.ImageAxes, 'Title')
            xlabel(app.ImageAxes, 'X')
            ylabel(app.ImageAxes, 'Y')
            zlabel(app.ImageAxes, 'Z')
            app.ImageAxes.Position = [36 221 585 460];

            % Create TraceAxes
            app.TraceAxes = uiaxes(app.ROIANALYSISUIFigure);
            title(app.TraceAxes, 'Title')
            ylabel(app.TraceAxes, 'Y')
            zlabel(app.TraceAxes, 'Z')
            app.TraceAxes.Position = [850 402 448 320];

            % Create RawAxes
            app.RawAxes = uiaxes(app.ROIANALYSISUIFigure);
            title(app.RawAxes, 'Title')
            xlabel(app.RawAxes, 'X')
            ylabel(app.RawAxes, 'Y')
            zlabel(app.RawAxes, 'Z')
            app.RawAxes.Position = [741 39 336 312];

            % Create CompareAxes
            app.CompareAxes = uiaxes(app.ROIANALYSISUIFigure);
            title(app.CompareAxes, 'Title')
            xlabel(app.CompareAxes, 'X')
            ylabel(app.CompareAxes, 'Y')
            zlabel(app.CompareAxes, 'Z')
            app.CompareAxes.Position = [1076 40 355 311];

            % Create LoadTIFButton
            app.LoadTIFButton = uibutton(app.ROIANALYSISUIFigure, 'push');
            app.LoadTIFButton.ButtonPushedFcn = createCallbackFcn(app, @LoadTIFButtonPushed, true);
            app.LoadTIFButton.BackgroundColor = [0 1 1];
            app.LoadTIFButton.FontSize = 18;
            app.LoadTIFButton.FontWeight = 'bold';
            app.LoadTIFButton.Position = [277 732 100 30];
            app.LoadTIFButton.Text = 'Load TIF';

            % Create DrawROIButton
            app.DrawROIButton = uibutton(app.ROIANALYSISUIFigure, 'push');
            app.DrawROIButton.ButtonPushedFcn = createCallbackFcn(app, @DrawROIButtonPushed, true);
            app.DrawROIButton.BackgroundColor = [0.902 0.902 0.902];
            app.DrawROIButton.FontSize = 18;
            app.DrawROIButton.FontWeight = 'bold';
            app.DrawROIButton.FontColor = [0 0 0];
            app.DrawROIButton.Position = [634 622 100 30];
            app.DrawROIButton.Text = 'Draw ROI';

            % Create AnalyzeButton
            app.AnalyzeButton = uibutton(app.ROIANALYSISUIFigure, 'push');
            app.AnalyzeButton.ButtonPushedFcn = createCallbackFcn(app, @AnalyzeButtonPushed, true);
            app.AnalyzeButton.BackgroundColor = [0.902 0.902 0.902];
            app.AnalyzeButton.FontSize = 18;
            app.AnalyzeButton.FontWeight = 'bold';
            app.AnalyzeButton.Position = [1320 650 100 30];
            app.AnalyzeButton.Text = 'Analyze';

            % Create SaveresultsButton
            app.SaveresultsButton = uibutton(app.ROIANALYSISUIFigure, 'push');
            app.SaveresultsButton.ButtonPushedFcn = createCallbackFcn(app, @SaveresultsButtonPushed, true);
            app.SaveresultsButton.BackgroundColor = [0.902 0.902 0.902];
            app.SaveresultsButton.FontSize = 18;
            app.SaveresultsButton.FontWeight = 'bold';
            app.SaveresultsButton.FontColor = [0 0 0];
            app.SaveresultsButton.Position = [1310 601 121 30];
            app.SaveresultsButton.Text = 'Save results';

            % Create FrameintervalsLabel
            app.FrameintervalsLabel = uilabel(app.ROIANALYSISUIFigure);
            app.FrameintervalsLabel.HorizontalAlignment = 'right';
            app.FrameintervalsLabel.FontSize = 18;
            app.FrameintervalsLabel.FontWeight = 'bold';
            app.FrameintervalsLabel.Position = [953 380 155 23];
            app.FrameintervalsLabel.Text = 'Frame interval (s)';

            % Create FramePeriodEditField
            app.FramePeriodEditField = uieditfield(app.ROIANALYSISUIFigure, 'numeric');
            app.FramePeriodEditField.FontSize = 18;
            app.FramePeriodEditField.FontWeight = 'bold';
            app.FramePeriodEditField.Position = [1123 380 100 24];
            app.FramePeriodEditField.Value = 0.2;

            % Create SetDefaultDirectoryButton
            app.SetDefaultDirectoryButton = uibutton(app.ROIANALYSISUIFigure, 'push');
            app.SetDefaultDirectoryButton.ButtonPushedFcn = createCallbackFcn(app, @SetDefaultDirectoryButtonPushed, true);
            app.SetDefaultDirectoryButton.BackgroundColor = [0 1 1];
            app.SetDefaultDirectoryButton.FontSize = 18;
            app.SetDefaultDirectoryButton.FontWeight = 'bold';
            app.SetDefaultDirectoryButton.Position = [36 732 194 30];
            app.SetDefaultDirectoryButton.Text = 'Set Default Directory';

            % Create DefaultDirLabel
            app.DefaultDirLabel = uilabel(app.ROIANALYSISUIFigure);
            app.DefaultDirLabel.WordWrap = 'on';
            app.DefaultDirLabel.FontSize = 14;
            app.DefaultDirLabel.FontAngle = 'italic';
            app.DefaultDirLabel.Position = [35 683 513 43];
            app.DefaultDirLabel.Text = '"Default directory: (not set)"';

            % Create BaselinecorrectionLabel
            app.BaselinecorrectionLabel = uilabel(app.ROIANALYSISUIFigure);
            app.BaselinecorrectionLabel.BackgroundColor = [0.9412 0.9412 0.9412];
            app.BaselinecorrectionLabel.HorizontalAlignment = 'right';
            app.BaselinecorrectionLabel.FontSize = 18;
            app.BaselinecorrectionLabel.FontWeight = 'bold';
            app.BaselinecorrectionLabel.FontColor = [0.149 0.149 0.149];
            app.BaselinecorrectionLabel.Position = [849 736 244 23];
            app.BaselinecorrectionLabel.Text = 'Baseline correction method';

            % Create BaselineMethodDropDown
            app.BaselineMethodDropDown = uidropdown(app.ROIANALYSISUIFigure);
            app.BaselineMethodDropDown.Items = {'None', 'Linear detrend', 'Exponential bleach', 'Whittaker baseline(AsLS)'};
            app.BaselineMethodDropDown.FontSize = 18;
            app.BaselineMethodDropDown.FontWeight = 'bold';
            app.BaselineMethodDropDown.FontColor = [0.149 0.149 0.149];
            app.BaselineMethodDropDown.BackgroundColor = [0.9412 0.9412 0.9412];
            app.BaselineMethodDropDown.Position = [1108 735 254 24];
            app.BaselineMethodDropDown.Value = 'None';

            % Create SaveROIButton
            app.SaveROIButton = uibutton(app.ROIANALYSISUIFigure, 'push');
            app.SaveROIButton.ButtonPushedFcn = createCallbackFcn(app, @SaveROIButtonPushed, true);
            app.SaveROIButton.BackgroundColor = [0.902 0.902 0.902];
            app.SaveROIButton.FontSize = 18;
            app.SaveROIButton.FontWeight = 'bold';
            app.SaveROIButton.FontColor = [0.149 0.149 0.149];
            app.SaveROIButton.Position = [634 547 100 30];
            app.SaveROIButton.Text = 'Save ROI';

            % Create LoadROIButton
            app.LoadROIButton = uibutton(app.ROIANALYSISUIFigure, 'push');
            app.LoadROIButton.ButtonPushedFcn = createCallbackFcn(app, @LoadROIButtonPushed, true);
            app.LoadROIButton.BackgroundColor = [0.902 0.902 0.902];
            app.LoadROIButton.FontSize = 18;
            app.LoadROIButton.FontWeight = 'bold';
            app.LoadROIButton.FontColor = [0.149 0.149 0.149];
            app.LoadROIButton.Position = [634 477 100 30];
            app.LoadROIButton.Text = 'Load ROI';

            % Create ClearROIButton
            app.ClearROIButton = uibutton(app.ROIANALYSISUIFigure, 'push');
            app.ClearROIButton.ButtonPushedFcn = createCallbackFcn(app, @ClearROIButtonPushed, true);
            app.ClearROIButton.BackgroundColor = [0.9412 0.9412 0.9412];
            app.ClearROIButton.FontSize = 18;
            app.ClearROIButton.FontWeight = 'bold';
            app.ClearROIButton.FontColor = [1 0 0];
            app.ClearROIButton.Position = [634 409 100 30];
            app.ClearROIButton.Text = 'Clear ROI';

            % Create ClearallButton
            app.ClearallButton = uibutton(app.ROIANALYSISUIFigure, 'push');
            app.ClearallButton.ButtonPushedFcn = createCallbackFcn(app, @ClearallButtonPushed, true);
            app.ClearallButton.BackgroundColor = [0 1 1];
            app.ClearallButton.FontSize = 18;
            app.ClearallButton.FontWeight = 'bold';
            app.ClearallButton.Position = [439 732 100 30];
            app.ClearallButton.Text = 'Clear all';

            % Create FrameSliderLabel
            app.FrameSliderLabel = uilabel(app.ROIANALYSISUIFigure);
            app.FrameSliderLabel.HorizontalAlignment = 'right';
            app.FrameSliderLabel.Enable = 'off';
            app.FrameSliderLabel.Position = [74 97 74 22];
            app.FrameSliderLabel.Text = 'Frame Slider';

            % Create FrameSlider
            app.FrameSlider = uislider(app.ROIANALYSISUIFigure);
            app.FrameSlider.Limits = [1 10];
            app.FrameSlider.ValueChangedFcn = createCallbackFcn(app, @FrameSliderValueChanged, true);
            app.FrameSlider.Enable = 'off';
            app.FrameSlider.Position = [170 106 403 3];
            app.FrameSlider.Value = 1;

            % Create PlayButton
            app.PlayButton = uibutton(app.ROIANALYSISUIFigure, 'push');
            app.PlayButton.ButtonPushedFcn = createCallbackFcn(app, @PlayButtonPushed, true);
            app.PlayButton.BackgroundColor = [0.9608 0.4667 0.1608];
            app.PlayButton.FontSize = 18;
            app.PlayButton.FontWeight = 'bold';
            app.PlayButton.Position = [278 164 100 30];
            app.PlayButton.Text = 'Play';

            % Create AdjustContrastButton
            app.AdjustContrastButton = uibutton(app.ROIANALYSISUIFigure, 'push');
            app.AdjustContrastButton.ButtonPushedFcn = createCallbackFcn(app, @AdjustContrastButtonPushed, true);
            app.AdjustContrastButton.FontSize = 18;
            app.AdjustContrastButton.FontWeight = 'bold';
            app.AdjustContrastButton.Position = [45 164 150 32];
            app.AdjustContrastButton.Text = 'Adjust Contrast';

            % Show the figure after all components are created
            app.ROIANALYSISUIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = GRAB_ROI_Analysis_v3

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.ROIANALYSISUIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.ROIANALYSISUIFigure)
        end
    end
end