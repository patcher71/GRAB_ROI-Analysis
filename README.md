# GRAB_ROI-Analysis
Analysis of in vitro GRAB fluorescence 


This program was designed with the help of ChatGPT to facilitate basic analysis of short video (TIF) files using GRAB fluorescent sensors in brain slices.

The workflow is fairly straighforward.  The user selects a root directory and an image file to open, and then draws an ROI surrounding the area of interest. The user has the option to import the whole file or a subset of frames for analysis.  The ROI can be saved so that multiple files from the same recording can be analyzed easily.  Once the ROI is drawn, the 'Analyze' button will plot both the mean intensity of the image as well as the deltaF/F0. 

Three baseline correction options are available in this version. This is useful because some degree of photobleaching always causes a downward trend over the course of the recording.

One is a simple linear detrend, the second correction option is a single exponential bleach decay, and the third is a more complex Whittaker baseline/asymmetric least squares algorithm for baselines that are a bit noisier or perhaps have multiple phases. This last model follows the slow drift of the baseline but 'de-emphasizes' the peak increases.  

Once the analysis is done, the user can save the results to an Excel workbook for further analysis, etc.


