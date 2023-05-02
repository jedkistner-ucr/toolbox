'''
Loads .npz datafiles that the builder module built and returns the data from them in numpy arrays
'''

import numpy as np
from os.path import join, exists

'''
Loads a set of data maps -- set up for Trevors data output style
Returns source drain, gate, photocurrent, and reflection
'''
def load_npzset(path, name):
    savefile = join(path,"data_Vsd_Vg",name+".npz")
    if exists(savefile):
        files = np.load(savefile)
        print(files.files)
        Vsd = files['Vsd']
        Vg = files['Vg']
        d = files['d']
        r = files['r']
    else:
        raise ValueError("Dataset " + str(name) + " file not found")

    return Vsd, Vg, d, r

'''
Loads a data map, e.g. dark current
Returns source drain, gate, and current
'''
def load_npzmap(path, name):
    savefile = join(path,"darkcurrent_Vsd_Vg",name+".npz")
    if exists(savefile):
        files = np.load(savefile)
        for f in files.iterkeys():
            print(f)
        Vsd = files['Vsd']
        Vg = files['Vg']
        pci = files['pci']
    else:
        raise ValueError(savefile+" not found")

    return Vsd, Vg, pci

'''
Loads single raw npy datamap
Needs location and x and y [start, end, steps]
Data array, x values and y values (essentially turns the data into a trevor type arrangement)
'''
def load_npymap(path, name, x, y):
    savefile = join(path,"darkcurrent_Vsd_Vg",name+".npz")
    if exists(savefile):
        data = np.load(savefile)
        xr = np.linspace(x[0], x[1], x[2])
        yr = np.linspace(y[0], y[1], y[2])
    else:
        raise ValueError(savefile+" not found")

    return data, xr, yr

'''
gets data from log file of 
single scan
Returns scan name, xtitle, ytitle, xrange, yrange, nx, ny, preamp, temperature, power
Needs a secondary power calculation module
'''
def get_info(path, name):
    savename = join(path, name + "_log.log")
    f = open(savename, "r")
    info = {}
    info["scan"] = name
    info["xrange"] = [0,0]
    info["yrange"] = [0,0]
    for line in f:
        s = line.split(':')

        if s[0] == "Fast Axis Variable":
            info["xname"] = s[1]
        elif s[0] == "Slow Axis Variable":
            info["yname"] = s[1]

        elif s[0] == "Fast Axis Start":
            info["xrange"][0] = float(s[1])
        elif s[0] == "Fast Axis End":
            info["xrange"][1] = float(s[1])
        
        elif s[0] == "Slow Axis Start":
            info["yrange"][0] = float(s[1])
        elif s[0] == "Slow Axis End":
            info["yrange"][1] = float(s[1])

        elif s[0] == "nx":
            info["nx"] = float(s[1])
        elif s[0] == "ny":
            info["ny"] = float(s[1])

        elif s[0] == "Pre-Amp Gain":
            info["gain"] = float(s[1])

        elif s[0] == "Temperature A":
            info["temp"] = float(s[1])

        elif s[0] == "Raw Power":
            info["power"] = float(s[1])

'''
loads basic jed type file
'''
def load_simplemap(path, name):
    savefile = join(path, name + ".npz")
    if exists(savefile):
        f = np.load(savefile, allow_pickle=True)

        d = f['d']
        xval = f['xval']
        yval = f['yval']

        return xval, yval, d


'''
loads jed type npz file
'''
def load_map(path, name, returnpower = False):
    savefile = join(path, name + ".npz")
    if exists(savefile):
        f = np.load(savefile, allow_pickle=True)

        if returnpower:
            d = f['d']
            xval = f['xval']
            yval = f['yval']
            power = f['pow']
            rfi = f['rfi']
            info = f['info']
            return xval, yval, d, rfi, power, info
        else:
            d = f['d']
            xval = f['xval']
            yval = f['yval']
            rfi = f['rfi']
            info = f['info']
            return xval, yval, d, rfi, info

def load_hypercube(path, name):
    savefile = join(path, name + ".npz")
    if exists(savefile):
        f = np.load(savefile, allow_pickle=True)

        d = f['data']
        xval = f['xval']
        yval = f['yval']
        cval = f['cval']
        zval = f['zval']
        rf = f['rfi']
        pow = f['pow']
        return xval, yval, cval, zval, d, rf, pow

def load_extracted(path, name):
    savefile = join(path, name + ".npz")
    if exists(savefile):
        f = np.load(savefile, allow_pickle=True)

        d = f['d']
        xval = f['xval']
        yval = f['yval']
        return xval, yval, d

def load_cube(path, name):
    savefile = join(path, name + ".npz")
    if exists(savefile):
        f = np.load(savefile, allow_pickle=True)

        d = f['d']
        xval = f['xval']
        yval = f['yval']
        cval = f['cval']
        rf = f['rfi']
        power = f['pow']
        info = f['inf']
        return xval, yval, cval, d, rf, power, info

#Just a simpler cube
def load_erfucube(path, name):
    savefile = join(path, name + ".npz")
    if exists(savefile):
        f = np.load(savefile, allow_pickle=True)

        d = f['d']
        xval = f['xval']
        yval = f['yval']
        cval = f['cval']
        return xval, yval, cval, d
    
def load_twobodyfit(path, name):
    savefile = join(path, name + ".npz")
    if exists(savefile):
        f = np.load(savefile, allow_pickle=True)

        alpha = f['alpha']
        beta = f['beta']
        cmap = f['cmap']
        alphaerror = f['alphaerror']
        betaerror = f['betaerror']
        cerror = f['cerror']
        noise = f['noise']
        yrange = f['yrange']
        xval = f['xval']
        yval = f['yval']
        rfi = f['rfi']
        info = f['info']

        return alpha, beta, cmap, alphaerror, betaerror, cerror, noise, yrange, xval, yval, rfi, info

def load_twobodyfit_cube(path, name):
    savefile = join(path, name + ".npz")
    if exists(savefile):
        f = np.load(savefile, allow_pickle=True)

        alpha = f['alpha']
        beta = f['beta']
        cmap = f['cmap']
        alphaerror = f['alphaerror']
        betaerror = f['betaerror']
        cerror = f['cerror']
        noise = f['noise']
        yrange = f['yrange']
        xval = f['xval']
        yval = f['yval']
        cval = f['cval']
        rfi = f['rfi']

        return alpha, beta, cmap, alphaerror, betaerror, cerror, noise, yrange, xval, yval, cval, rfi

def load_9gag(path, name):

    #read in full list
    f = open(join(path, name), 'r')
    data = f.readlines()

    #[Author, Days since show drop, image category, post title, post id]
    title = []
    author = []
    date = []
    cat = []
    postID = []

    cats = data[1].split(",")
    cats[-1] = cats[-1][:-1]

    for i in range(len(data)-2):
        line = data[i+2].split(",:,")
        author.append(line[0])
        date.append(int(line[1]))
        cat.append(line[2])
        title.append(line[3])
        postID.append(line[4])

    f.close()

    return author, date, cat, title, postID, cats

def load_reddit_analytics(path, name):

    #read in full list
    f = open(join(path, name), 'r')
    data = f.readlines()

    #[Author, Days since show drop, image category, post title, post id]
    authors = []
    posts = []

    for i in range(len(data)):
        line = data[i].split(":")
        authors.append(line[0])
        line[-1] = line[-1][:-1]

        dataline = [[],[],[],[],[]]
        for p in range(len(line)-1):
            holding = []
            if line[1+p] != '':
                d = line[1+p].split(",")
                for dp in d:
                    holding.append(dp)
            dataline[p] = holding
        posts.append(dataline)

    f.close()
    
    return authors, posts