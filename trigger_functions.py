import numpy as np
import matplotlib.pyplot as plt

def gaussian_convergence(data, threshold = 5, attempts = 5):
    data_conv = data.copy()
    convergence = 0
    for k in range(attempts):
        mean = np.mean(data_conv)
        std = np.std(data_conv)
        
        bool_test1 = data_conv > (mean + threshold*std)
        bool_test2 = data_conv < (mean - threshold*std)
        
        bool_test = bool_test1 + bool_test2
        
        if any(bool_test):
            data_conv = data_conv[~bool_test]
        else:
            #convergence = True
            break
        convergence += 1
    return data_conv, convergence

def find_stdArray(img_arr, divisions, overlap = 33):
    
    n_evs = np.shape(img_arr)[0]
    size = np.shape(img_arr)[1]
    
    std_arr = np.zeros([n_evs, divisions[0]*divisions[1]], dtype = float)
    
    sub_size = int((size/divisions[0]) + overlap)
    stride = int((size/divisions[0]) - (overlap/(divisions[0]-1))) 
    
    for ev in range(n_evs):
        img = img_arr[ev]

        x, y = np.meshgrid(np.arange(divisions[0]), np.arange(divisions[1]))

        #print("Image: %.3f \u00B1 %.3f" %(np.mean(img.flatten()), np.std(img.flatten())))
        for k in range(divisions[0]*divisions[1]):
            xx = x.flatten()[k]
            yy = y.flatten()[k]

            x_begin = xx * stride
            x_end = x_begin + sub_size

            y_begin = yy * stride
            y_end = y_begin + sub_size


            img_div = img[x_begin:x_end, y_begin:y_end]
            std_arr[ev,k] = np.std(img_div)

    return std_arr

def find_threshold(std_arr, threshold = 5):
    
    div = np.shape(std_arr)[1]
    threshold_arr = np.zeros(div, dtype = float)
    
    for k in range(div):
        hist_arr_plot = std_arr[:,k]
        data_conv, convergence = gaussian_convergence(hist_arr_plot)
        mu = np.mean(data_conv)
        sigma = np.std(data_conv)

        threshold_arr[k] = mu+(threshold*sigma)
        #print(convergence)

    return threshold_arr

def trigger(std_arr, threshold):
    
    trigger_arr = std_arr > threshold
    return np.where(trigger_arr)[0]

def plot_image(bank, vmin, vmax, grid, event_number, trigger_position, divisions, y0=100, overlap = 33):

    shape = int(np.sqrt(bank.size_bytes*8/16))
    image = np.reshape(bank.data, (shape, shape))
    
    fig, ax = plt.subplots(1, 2, figsize=(24,12))
    ax[0].imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    ax[0].set_title("Event I%d" %(event_number))
    
    if grid:
        ax[0] = plt.gca();

        majorx_ticks = np.arange(0, shape,  int(shape/11))
        majory_ticks = np.arange(0, shape,  int(shape/11))
        # Major ticks
        ax[0].set_xticks(majorx_ticks)
        ax[0].set_yticks(majory_ticks)

        # Labels for major ticks
        ax[0].set_xticklabels(majorx_ticks)
        ax[0].set_yticklabels(majory_ticks)

        ax[0].grid(color='r', linestyle='--', linewidth=1)

        ax[0].hlines(y0, 0, shape-1, colors='g')
        ax[0].hlines(shape-y0, 0, shape-1, colors='g')
        ax[0].vlines(y0, 0, shape-1, colors='g')
        ax[0].vlines(shape-y0, 0,shape-1, colors='g')
    
    size = np.shape(image)[0]
    sub_size = int((size/divisions[0]) + overlap)
    stride = int((size/divisions[0]) - (overlap/(divisions[0]-1))) 
    
    x, y = np.meshgrid(np.arange(divisions[0]), np.arange(divisions[1]))
    
    img_trigger = np.zeros([size,size])
    
    if trigger_position.any():
        for k in trigger_position:
            xx = x.flatten()[k]
            yy = y.flatten()[k]

            x_begin = xx * stride
            x_end = x_begin + sub_size

            y_begin = yy * stride
            y_end = y_begin + sub_size
            
            img_trigger[x_begin:x_end, y_begin:y_end] = image[x_begin:x_end, y_begin:y_end]
    
    ax[1].imshow(img_trigger, cmap='gray', vmin=vmin, vmax=vmax)
    ax[1].grid(color='gray', linestyle='--', linewidth=1)
    ax[1].set_xticks(np.arange(0,2304,2304/12))
    ax[1].set_yticks(np.arange(0,2304,2304/12))
    ax[1].set_title("Event I%d" %(event_number))

    plt.show()
    
    return