'''
Modules that take raw data files and compile them into a single .npz datafile
'''

import numpy as np
from os.path import join, exists
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as pl
from scipy import ndimage

def main():
    path = "C:/Users/Jedki/QMOdata/Raw Data/2020_08/2020_08_04"
    newpath = "C:/Users/jedki/QMOdata/Built"
    name = "2019_09_24_118"
    newname = "2020_08_04_cuts"
    #dark current scans
    # names = ["2019_09_25_231","2019_09_24_48","2019_09_24_58","2019_09_24_73","2019_09_24_84","2019_09_24_97","2019_09_24_112","2019_09_25_10", "2019_09_25_28" ,"2019_09_25_58","2019_09_25_82","2019_09_25_119"]#"2019_09_25_232","2019_09_25_233","2019_09_25_234","2019_09_25_235","2019_09_25_236","2019_09_25_237","2019_09_25_238","2019_09_25_239","2019_09_25_240"]

    names = []
    for i in range(12):
        number = 35 + i
        names.append("CPS_2020_08_03_" + str(number))
    for i in range(39):
        number = 0 + i
        names.append("CPS_2020_08_04_" + str(number))


    #fixed bright current scans
    # names = [ "2019_09_25_204","2019_09_24_54","2019_09_24_67","2019_09_24_80","2019_09_24_91","2019_09_24_107","2019_09_24_118","2019_09_25_17","2019_09_25_46","2019_09_25_73","2019_09_25_104","2019_09_25_138"]
    # build_map(path, name, newpath, False, 1)
    build_superset(path, names, newname, newpath)

def line(x, m, b):
    i = m*x + b
    return i

'''
does power fit and return power value or values
'''
def powernorm(sensorsave, wavelength, values, calwave):
    wavesave = "C:/Users/Gutzz/Desktop/Data/Calibration/InGaAs/PDA20CS_responsivity.txt"
    wavefile = join(wavesave)
    sensorfile = join(sensorsave)
    #first we need the multiplier from the sensor calibration via wavelength (A/W)
    #Calwave is the wavelength that the sensor is calibrated at
    f = np.loadtxt(wavefile)
    f = np.transpose(f)

    func = interp1d(f[0,:], f[1,:])
    normval = func(calwave)

    f[1,:] = f[1,:] / normval

    func = interp1d(f[0,:], f[1,:])
    norm = 1 / func(wavelength)

    #SANITY CHECK PLOTS
    # pl.figure()
    # pl.plot(f[0,:], f[1,:] * normval)
    # pl.plot(f[0,:], func(f[0,:]))
    # pl.show()

    #Now we need to get the sensor data and fit a line to it. The sensor data is multiplied by a factor from the wavelength
    f = np.loadtxt(sensorfile)
    f = np.transpose(f)
    
    func = interp1d(f[1,:], norm * f[0,:])

    #Added to catch odd power values that might be below the acceptable minimum
    interp_min = np.min(f)
    for x in range(values[:,0].size):
        for y in range(values[0,:].size):
            if values[y,x] < interp_min:
                values[y,x] = interp_min + .0001

    newval = func(values)

    #SANITY CHECK PLOTS
    # x = np.linspace(.11,5,20)
    # pl.figure()
    # pl.plot(x, func(x))
    # pl.plot(f[1,:], f[0,:])
    # pl.show()
    return newval

'''
takes a 2d scan and builds all the data into an npz file with corrected power and current values
'''
def build_map(path, name, newpath, gaussfilter = False, filterval = 1, superset = False):
    sensorfile = "C:/Users/Gutzz/Desktop/Data/Calibration/Power/2019_09_04_data.txt"
    datafile = join(path, name + "_pci.dat")
    logfile = path + "/" + name + "_log.log"
    powfile = join(path, name + "_pow.dat")
    rfifile = join(path, name + "_rfi.dat")
        
    ### LOG FILE LOAD ###
    f = open(logfile, "r")
    info = {}
    info["scan"] = name

    lc = 0
    for line in f:
        s = line.split(':')

        info[s[0]] = s[1]

        lc += 1
        if lc > 39:
            break

        # if s[0] == "Fast Axis Variable":
        #     info["xname"] = s[1]
        # elif s[0] == "Slow Axis Variable":
        #     info["yname"] = s[1]
        

        # elif s[0] == "Fast Axis Start":
        #     xr[0] = float(s[1])
        # elif s[0] == "Fast Axis End":
        #     xr[1] = float(s[1])
        
        # elif s[0] == "Slow Axis Start":
        #     yr[0] = float(s[1])
        # elif s[0] == "Slow Axis End":
        #     yr[1] = float(s[1])

        # elif s[0] == "nx":
        #     nx = int(s[1])
        #     info["nx"] = nx
        # elif s[0] == "ny":
        #     ny = int(s[1])
        #     info["ny"] = ny
        # elif s[0] == "Line Rate":
        #     lr = float(s[1])
        #     info["linerate"] = lr

        # elif s[0] == "Source/Drain Start":
        #     info["sd0"] = float(s[1])  
        # elif s[0] == "Source/Drain End":
        #     info["sd1"] = float(s[1])  

        # elif s[0] == "Backgate Start":
        #     info["gate0"] = float(s[1])            
        # elif s[0] == "Backgate End":
        #     info["gate1"] = float(s[1])

        # elif s[0] == "Pre-Amp Gain":
        #     info["gain"] = float(s[1])
        # elif s[0] == "Lock-In Gain":
        #     info["lockin"] = float(s[1])

        # elif s[0] == "Temperature A":
        #     info["temp"] = float(s[1])

        # elif s[0] == "Raw Power":
            # info["power"] = float(s[1])

        # elif s[0] == "Wavelength":
            # info["wave"] = float(s[1])
        
    # Power correction!
    # for MDQ poewr is always going to be zero
    # info["power"] = 0
    xval = np.linspace(float(info["Fast Axis Start"]), float(info["Fast Axis End"]), int(info["nx"]))
    yval = np.linspace(float(info["Slow Axis Start"]), float(info["Slow Axis End"]), int(info["ny"]))

    rawpower = np.loadtxt(powfile)
    # averawpower = np.mean(rawpower)
    # if averawpower < .102:
    #     info["power"] = 0
    #     power = np.zeros(rawpower.shape)
    # else:
        #8/4/2020 power will be wrong as missing wave file       
        # power = rawpower#powernorm(sensorfile, info["wave"], rawpower, 1250)
        # avepower = np.mean(power)
        # info["power"] = avepower
    # End power correction!

    # Converts dictionary to array
    inf = np.empty((2,len(info)), dtype='object')
    i = 0
    for k in info:
        inf[0][i] = k
        inf[1][i] = info[k]
        i = i+1

    ### END LOG FILE LOAD ###

    ### START DATA LOAD ###

    data = np.loadtxt(datafile)

    if gaussfilter:
        data = ndimage.gaussian_filter(data, filterval)

    gain = float(info["Pre-Amp Gain"]) #*.001
    data = gain * data
    # if avepower > 

    power = rawpower

    rf = np.loadtxt(rfifile)


    ### END DATA LOAD ###

    print(inf)
    if superset:
        return data, xval, yval, power, rf, inf, info
    else:
        savename = join(newpath, name)
        np.savez(savename, d = data, xval = xval, yval = yval, rfi = rf, pow = power, info = inf)   

'''
Takes a set of data files and buids a 4 dimensional cube
'''
def build_superset(path, names, newsave, newpath):
    d = []
    x = []
    y = []
    p = []
    r = []
    inf0 = []
    # temps = []
    powers = []
    for name in names:
        d1, x1, y1, p1, r1, i1 , id1= build_map(path, name, "", superset=True)
        d.append(d1)
        x.append(x1)
        y.append(y1)
        p.append(p1)
        r.append(r1)
        inf0.append(i1)
        powers.append(id1['power'])

    data = np.asarray(d)
    xval = np.asarray(x)
    yval = np.asarray(y)
    power = np.asarray(p)
    rf = np.asarray(r)
    inf = np.asarray(inf0)

    print(str(data.shape))
    print(str(xval.shape))
    print(str(inf.shape))

    print(powers)
    powers = np.asarray(powers)
    powers = np.sort(powers)
    print(powers)


    savename = join(newpath, newsave)
    np.savez(savename, d = data, xval = xval, yval = yval, power = power, rfi = rf, info = inf)   

#At this point we have them all stacked
# - Convert to arrays
# - Interpolate to get data on same pixels



if __name__ == '__main__':
    main()
