# toolbox
Scripts for plot generation and set building

I drop these in the python site-packages folder so i can access them from scripts running anywhere on my computer without having to add to my path variable.

Builder.py
  Primarily contains a method that takes the pci, rfi, pow, and log.log files from a simple 2d scan and compines them into a single .npz file. This is a bit outdated
  as i have other scripts that compile 3d and 4d datasets in the same way but better and as thier own files
  
Loader.py
  Contains scripts that unpack all the different .npz files and return the data arrays from them
  
AnalysisTools.py
  Contains a few small methods that i thought i would often use. Essentially completely obselete at this point
  
PlotTools.py
  By far the most useful of all the files. Contains code that produces a variety of forms and shapes of figures
  
CustomMaps.py
  The same as plottools except i primarily keep complex figure plots in custom maps and more versatile but visually simple plots in plottools
