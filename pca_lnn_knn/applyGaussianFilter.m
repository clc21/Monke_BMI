function smoothedSpikeRate = applyGaussianFilter(spikeRate, sigma, winSz)
arguments
    spikeRate
    sigma double = 30  % Standard deviation for Gaussian smoothing (in ms)
    winSz double = 10  % Window size in ms
end
    % Create Gaussian kernel
    kernelSize = ceil(3 * sigma / winSz) * 2 + 1;  % Ensure it's odd
    t = linspace(-kernelSize / 2, kernelSize / 2, kernelSize);
    gaussKernel = exp(-t.^2 / (2 * sigma^2));
    gaussKernel = gaussKernel / sum(gaussKernel);  % Normalize

    % Apply convolution along the time dimension (2nd axis)
    smoothedSpikeRate = conv2(spikeRate, gaussKernel, 'same');
end