% Script for post-processing and visualizing suction-based grasping
% affordance predictions
%function location_2d = visualize()
% User options (change me)
backgroundColorImage = 'demo/back_color.png';   % 24-bit RGB PNG
backgroundDepthImage = 'demo/back_depth.png';   % 16-bit PNG depth in deci-millimeters
inputColorImage = 'demo/0000_color.png';             % 24-bit RGB PNG
inputDepthImage = 'demo/0000_depth.png';             % 16-bit PNG depth in deci-millimeters
cameraIntrinsicsFile = 'demo/test-camera-intrinsics.txt';  % 3x3 camera intrinsics matrix
resultsFile = 'demo/results.h5';                           % HDF5 ConvNet output file from running infer.lua

% Read RGB-D images and intrinsics
backgroundColor = img_full_3c(double(imread(backgroundColorImage))./255);
backgroundDepth = img_full(double(imread(backgroundDepthImage))./10000);
inputColor = img_full_3c(double(imread(inputColorImage))./255);
inputDepth = img_full(double(imread(inputDepthImage))./10000);
cameraIntrinsics = dlmread(cameraIntrinsicsFile);

% Read raw affordance predictions
results = hdf5read(resultsFile,'results');
results = permute(results,[2,1,3,4]); % Flip x and y axes
affordanceMap = results(:,:,2); % 2nd channel contains positive affordance
affordanceMap = imresize(affordanceMap,size(inputDepth)); % Resize output to full  image size 

% Clamp affordances back to range [0,1] (after interpolation from resizing)
affordanceMap(affordanceMap >= 1) = 0.9999;
affordanceMap(affordanceMap < 0) = 0;

% Post-process affordance predictions and generate surface normals
% extend 

[affordanceMap,surfaceNormalsMap] = postprocess(affordanceMap, ...
                                    inputColor,inputDepth, ...
                                    backgroundColor,backgroundDepth, ...
                                    cameraIntrinsics);

% Gaussian smooth affordances
affordanceMap = imgaussfilt(affordanceMap, 7);
afmap = affordanceMap
afmax = max(max(afmap))
location = find(afmap == afmax)
location_2d = [floor(location/480)+1,mod(location,480)]
% Generate heat map visualization for affordances
cmap = jet;
affordanceMap = cmap(floor(affordanceMap(:).*size(cmap,1))+1,:);
affordanceMap = reshape(affordanceMap,size(inputColor));

% Overlay affordance heat map over color image and save to results.png
figure(1); imshow(0.5*inputColor+0.5*affordanceMap);
figure(2); imshow(surfaceNormalsMap);
imwrite(0.5*inputColor+0.5*affordanceMap,'results.png')
imwrite(surfaceNormalsMap,'normals.png')
%end
