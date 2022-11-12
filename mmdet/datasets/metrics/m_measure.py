import numpy as np
from scipy import ndimage

"""
def calcMse(imgId):
    ##  Crop images if enabled
    if args.roi is not None:
        roi = args.roi

        if args.roi.find(',') != -1:
            roi = roi.split(',')
            p0 = [int(roi[0].split(':')[1]),int(roi[0].split(':')[0])]
            p1 = [int(roi[1].split(':')[1]),int(roi[1].split(':')[0])]
        
            print('Cropping rectangular ROI with vertices:  ( ', \
                    p0[0],' , ', p0[1], ')   ( ', p1[0],' , ',p1[1], ')')
        
            image1 = proc.crop_image( image1 , p0 , p1 )
            image2 = proc.crop_image( image2 , p0 , p1 )

        else:
            print('\nUsing pixels specified in file:\n', roi)
            file_roi = open( roi , 'r' )
            pixels = np.loadtxt( file_roi )
            image1 = image1[pixels[:,0],pixels[:,1]]
            image2 = image2[pixels[:,0],pixels[:,1]]
            num_pix = len( image1 )
            fact = factors( num_pix )
            image1 = image1.reshape( fact[0] , int( num_pix/fact[0] ) )
            image2 = image2.reshape( fact[0] , int( num_pix/fact[0] ) )   


    ##  Compute the gradient of the images, if enabled
    if args.gradient is True:
        image1 = compute_gradient_image( image1 )
        image2 = compute_gradient_image( image2 )      


    ##  Check whether the 2 images have the same shape
    if image1.shape != image2.shape:
        sys.error('\nERROR: The input images have different shapes!\n')


    ##  Plot to check whether the images have the same orientation
    if args.plot is True:
        print('\nPlotting images to check orientation ....')
        img_list = [ image1 , image2 ]
        title_list = [ 'Oracle image' , 'Image to analyze' ]
        dis.plot_multi( img_list , title_list , 'Check plot' )    


    ##  Compute figures of merit
    SNR = calc_snr( image1 , image2 )
    PSNR = calc_psnr( image1 , image2 )
    RMSE = calc_rmse( image1 , image2 )
    MAE = calc_rmse( image1 , image2 ) 

    results.append( np.array( [ SNR , PSNR , RMSE , MAE ] ) )
"""

def complexity_struct_info( image ):
    print('\n2) Calculate image complexity index based on spatial information ....')  
    
    ##  Applying vertical and horizontal Sobel filter
    print('Applying vertical and horizontal Sobel filter')    
    dy = ndimage.filters.sobel( image , 1 )
    dx = ndimage.filters.sobel( image , 0 ) 

    ##  Calculate the map of the spatial information (SI) or,
    ##  in other words, the map of magnitudes of the edge images
    npix   = image.shape[0] * image.shape[1] 
    map_si =  dx*dx + dy*dy

    si_mean = np.mean( map_si )
    si_rms  = np.sqrt( 1.0/npix * np.sum( map_si * map_si )  )
    si_std  = np.std( map_si )

    print('\nSI -- mean: ', si_mean )
    print('SI -- rms: ', si_rms ) 
    print('SI -- std: ', si_std )     

    ##  Calculate the gradient sparsity index ( GSI )
    map_si = map_si != 0
    complexity = np.count_nonzero( map_si ) / myfloat( npix ) 
    
    print('\n3) Calculate image complexity index based on gradient sparsity index ....')     
    print('\nGradient sparsity index: ', complexity)
    print('\n')