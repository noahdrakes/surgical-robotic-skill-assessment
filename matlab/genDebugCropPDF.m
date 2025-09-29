function genDebugCropPDF(debugDir)

    if nargin < 1 || ~isfolder(debugDir)
        error('Please provide a valid debugDir path');
    end

    if ~isfolder(fullfile(debugDir, 'PDF'))
        mkdir(fullfile(debugDir, 'PDF'));
    end

    function createPDFfromPNGs(filePattern, pdfName)
        pngFiles = dir(fullfile(debugDir, 'PNG', filePattern));
        if isempty(pngFiles)
            fprintf('No files found for pattern: %s\n', filePattern);
            return;
        end

        nImages = length(pngFiles);
        imagesPerPage = 4;
        nPages = ceil(nImages / imagesPerPage);

        fig = figure('Visible', 'off', 'Units', 'normalized', 'Position', [0 0 1 1]);

        for p = 1:nPages
            clf(fig);
            for i = 1:imagesPerPage
                idx = (p-1)*imagesPerPage + i;
                if idx > nImages
                    break;
                end
                imgPath = fullfile(debugDir, 'PNG', pngFiles(idx).name);
                img = imread(imgPath);
                subplot(2, 2, i);
                imshow(img);
                axis off;
                axis image;
            end

            pdfPath = fullfile(debugDir, 'PDF', pdfName);
            if p == 1
                exportgraphics(fig, pdfPath, 'Append', false);
            else
                exportgraphics(fig, pdfPath, 'Append', true);
            end
        end

        close(fig);

        fprintf('✔ Created %s from %d images\n', pdfName, nImages);
    end

    % Generate both PDFs
    createPDFfromPNGs('*_cropButtonPress.png', 'DebugCropButtonPress.pdf');
    createPDFfromPNGs('*_cropTrialEnd.png', 'DebugCropTrialEnd.pdf');
end