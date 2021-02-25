function board_position(row,col)

    % Load Image
    im = imread('chess.png');
    imshow(im)
    
    % Obtain the size of the image
    [rows, columns] = size(im);
    
    % Calculate the board size 
    board_size = rows/8;  %total 8 
    
    % Calculate the require position
    row_start = row * board_size + 1; %begin with 0
    col_start = col * board_size + 1;
    
    row_end = row_start + board_size - 1;
    col_end = col_start + board_size - 1;
    
    % Crop region starting at (100,120)
    im_crop = im(row_start : row_end,col_start:col_end,:);

    im_gray = rgb2gray(im_crop);
    
    % Detect how many zero(white) values in the image
    n = nnz(im_gray); %返回矩阵X中非零元素的个数
    m = numel(im_gray); %返回元素个数
    
    if all(im_gray(:) == im_gray(1))
        disp('Empty board');
    else
        white_pixel = (n/m) * 100;
        
        if white_pixel > 80
            disp('white piece');
        else
            disp('black piece');
        end
    end
    
    % Display the croped image
    imshow(im_gray)

end
