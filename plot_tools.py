'''
Holds plot generation scripts and data color modules
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl

'''
what do we do everytime we want to display data?
'''
#Make colormap with min and max normalization based on data min and max
def color_memap(colorname, data, dmin = 0, dmax = 0, invertz = False):
    cmap = pl.get_cmap(colorname)
    if dmin == dmax:
        if invertz:
            cnorm = mpl.colors.Normalize(vmin = np.max(data), vmax = np.min(data))
        else:
            cnorm = mpl.colors.Normalize(vmin = np.min(data), vmax = np.max(data))
    else:
        if invertz:
            cnorm = mpl.colors.Normalize(vmin = dmax, vmax = dmin)
        else:
            cnorm = mpl.colors.Normalize(vmin = dmin, vmax = dmax)
    return cmap, cnorm

#Get color list
def color_meline(colorname, no):
    cmap = pl.get_cmap(colorname, no)
    clr = cmap(range(no))
    return clr

#Get extent for imshow based on min and maxes of data xrange and yrange
def get_extent(xr, yr, inverty = False, invertx = False):
    xmin = np.min(xr)
    xmax = np.max(xr)
    ymin = np.min(yr)
    ymax = np.max(yr)
    if inverty and not invertx:
        extent=[xmin, xmax, ymin, ymax]
    elif invertx and not inverty:
        extent=[xmax, xmin, ymax, ymin]
    elif invertx and inverty:
        extent=[xmax, xmin, ymin, ymax]
    else:    
        extent=[xmin, xmax, ymax, ymin]
    return extent

#Turns the info array back into a dictionary
def make_dict(info):
    keys = {}
    for i in range(info[0,:].size):
        keys[str(info[0,i])] = info[1,i]
    return keys

'''
Returns axes rectangles for a given number of plots and scale
'''
def maketriple(plots, scale = 1):
    #Standard placements in inches
    colorbarwidth = .25 * scale
    colorbaroff = .125 * scale
    plotseperation = 1 * scale
    figuremargin = 1 * scale
    plotwidth = 4 * scale
    #Calculates figuresize
    figuresize = ((plotwidth * plots) + colorbaroff + colorbarwidth + (figuremargin * 2) + (plotseperation * (plots - 1)) , plotwidth + (2 * figuremargin))

    #placements for inline colorbars
    insetcoloroff = [.15 * scale, .01 * scale]
    insetcolorwidth = [1.3 * scale,.15 * scale]

    #Converts to percents
    inclro = [ insetcoloroff[0] /  figuresize[0] , insetcoloroff[1] /  figuresize[1] ]
    inclrw = [ insetcolorwidth[0] / figuresize[0] , insetcolorwidth[1] / figuresize[1] ]

    clrw = colorbarwidth / figuresize[0]
    clro = colorbaroff / figuresize[0]
    plotsep = plotseperation / figuresize[0]
    figurm = figuremargin / figuresize[0]
    figurmh = figuremargin / figuresize[1]
    plotw = plotwidth / figuresize[0]
    ploth = plotwidth / figuresize[1]

    #Defines rectangles for map and its colorbar
    ax1 = [ figurm, figurmh, plotw, ploth ]
    ax2 = [ax1[0] + ax1[2] + clro, figurmh, clrw, ploth ]

    #initiate axes lists
    ax = []
    axcb = [ax2]    #map colorbar

    #generate and append axes for linecut graphs
    for i in range(plots - 1):
        if i == 0:
            ax.append([ ax2[0] + ax2[2] + clrw + plotsep, figurmh, plotw, ploth ])
        else:
            ax.append([ ax[i-1][0] + ax[i-1][2] + plotsep, figurmh, plotw, ploth ])
    
    #generate and append inline colorbar axes
    for i in range(len(ax)):
        axcb.append([ ax[i][0] + inclro[0], ax[i][1] + ploth - inclro[1] - inclrw[1], inclrw[0], inclrw[1] ])

    #Insert map axes into first position
    ax.insert(0, ax1)

    return ax, axcb, figuresize

'''
the above makes plots following a 2d map, this just makes single plots
'''
def makeaxes(plots, cbar = False, scalebar = False, scale = 1):
    #in inches
    plotseperation = 1.4 * scale
    figuremargin = 1 * scale
    plotwidth = 4 * scale
    if cbar:
        colorbarwidth = .25 * scale
        colorbaroff = .125 * scale
    else:
        colorbarwidth = 0
        colorbaroff = 0
    if scalebar:
        scalebarwidth = 1 * scale
        scalebarh = .1 * scale
        scalebaroff = .1 * scale
    else:
        scalebarwidth = 0
        scalebarh = 0
        scalebaroff = 0
    #Calculates figuresize
    figuresize = ((plotwidth * plots) + (figuremargin * 3.8)+ ((colorbaroff + colorbarwidth) * (plots - 1) )+ (plotseperation * (plots - 1)) , plotwidth + (2 * figuremargin))

    #converting to percents
    plotsep = plotseperation / figuresize[0]
    figurm = figuremargin / figuresize[0]
    figurmh = figuremargin / figuresize[1]
    plotw = plotwidth / figuresize[0]
    ploth = plotwidth / figuresize[1]
    clrw = colorbarwidth / figuresize[0]
    clro = colorbaroff / figuresize[0]
    sclbw = scalebarwidth / figuresize[0]
    sclbow = scalebaroff / figuresize[0]
    sclboh = scalebaroff / figuresize[1]
    sclbh = scalebarh / figuresize[1]

    ax0 = [figurm + .5 * figurm, figurmh, plotw, ploth]
    ax = [ax0]

    if plots > 1:
        for i in range(plots):
            if i > 0:
                ax.append([ ax[i - 1][0] + plotw + plotsep + clrw + clro, figurmh, plotw, ploth ])

    axcb = []
    if cbar:
        for  a in ax:
            axcb.append([ a[0] + plotw + clro, figurmh, clrw, ploth ])   
    axscb = []
    if scalebar:
        for i in range(len(ax)):
            if i > 0:
                axscb.append([ax[i][0] + sclbow, ploth + figurmh - sclboh - sclbh, sclbw, sclbh])


    if cbar and scalebar:
        return ax, axcb, axscb, figuresize
    elif cbar:
        return ax, axcb, figuresize
    elif scalebar:
        return ax, axscb, figuresize
    else:
        return ax, figuresize

'''
makes a number of maps with colormaps
'''

def makemaps(plots, scale = 1):
    #in inches
    plotseperation = 1.4 * scale
    figuremargin = .6 * scale
    plotwidth = 4 * scale

    colorbarwidth = .25 * scale
    colorbaroff = .125 * scale

    #Calculates figuresize
    figuresize = ( ((plotwidth + colorbaroff + colorbarwidth) * plots) + ((plotseperation) * (plots -1)) + (figuremargin * 3) , plotwidth + (2 * figuremargin))

    #converting to percents
    plotsep = plotseperation / figuresize[0]
    figurm = figuremargin / figuresize[0]
    figurmh = figuremargin / figuresize[1]
    plotw = plotwidth / figuresize[0]
    ploth = plotwidth / figuresize[1]
    clrw = colorbarwidth / figuresize[0]
    clro = colorbaroff / figuresize[0]


    ax = []
    axcb = []
    
    for i in range(plots):
        left, bottom = figurm + (i * (plotw + clrw + clro + plotsep)), figurmh
        ax.append([left, bottom, plotw, ploth])
        axcb.append([left + plotw + clro, bottom, clrw, ploth])

    return ax, axcb, figuresize


'''
the above makes plots following a 2d map, this just makes single plots
'''
def makeaxesarray(plots, rows, cbar = False, scalebar = False, scale = 1):
    
    horizontalsep = 1.5 * scale
    plotseperation = 1.4 * scale
    figuremargin = 1 * scale
    plotwidth = 4 * scale
    if cbar:
        colorbarwidth = .25 * scale
        colorbaroff = .125 * scale
    else:
        colorbarwidth = 0
        colorbaroff = 0
    if scalebar:
        scalebarwidth = 1 * scale
        scalebarh = .1 * scale
        scalebaroff = .1 * scale
    else:
        scalebarwidth = 0
        scalebarh = 0
        scalebaroff = 0
    #Calculates figuresize
    figuresize = ((plotwidth * plots) + (figuremargin * 3.8)+ ((colorbaroff + colorbarwidth) * (plots - 1) )+ (plotseperation * (plots - 1)) ,
     (rows * plotwidth) + (2 * figuremargin) + (rows - 1) * horizontalsep)

    #converting to percents
    plotsep = plotseperation / figuresize[0]
    figurm = figuremargin / figuresize[0]
    figurmh = figuremargin / figuresize[1]
    plotw = plotwidth / figuresize[0]
    ploth = plotwidth / figuresize[1]
    clrw = colorbarwidth / figuresize[0]
    clro = colorbaroff / figuresize[0]
    sclbw = scalebarwidth / figuresize[0]
    sclbow = scalebaroff / figuresize[0]
    sclboh = scalebaroff / figuresize[1]
    sclbh = scalebarh / figuresize[1]
    horizsep = horizontalsep / figuresize[1]

    ax_master = []
    axcb_master = []
    axscb_master = []

    for p in range(rows):

        horizshift = p * (ploth + horizsep)

        ax0 = [figurm + .5 * figurm, figurmh + horizshift, plotw, ploth]
        ax = [ax0]

        if plots > 1:
            for i in range(plots):
                if i > 0:
                    ax.append([ ax[i - 1][0] + plotw + plotsep + clrw + clro, figurmh + horizshift, plotw, ploth ])

        axcb = []
        if cbar:
            for a in ax:
                axcb.append([ a[0] + plotw + clro, figurmh + horizshift, clrw, ploth ])   
        axscb = []
        if scalebar:
            for i in range(len(ax)):
                if i > 0:
                    axscb.append([ax[i][0] + sclbow, ploth + figurmh - sclboh - sclbh + horizshift, sclbw, sclbh])

        ax_master.append(ax)
        axcb_master.append(axcb)
        axscb_master.append(axscb)

    if cbar and scalebar:
        return ax_master, axcb_master, axscb_master, figuresize
    elif cbar:
        return ax_master, axcb_master, figuresize
    elif scalebar:
        return ax_master, axscb_master, figuresize
    else:
        return ax_master, figuresize

def makemapgrid(rows, cols, scale = 1, colorbar = False):
    #in inches
    figuremargin = 1 * scale
    plotwidth = 2 * scale
    cbarmargin_ = .1
    cbarwidth_ = .2

    figuresize = ( (figuremargin * 2) + (cols * plotwidth) + cbarwidth_ + cbarmargin_, (figuremargin * 2) + (rows * plotwidth) )

    #in percents
    figurm = figuremargin / figuresize[0]
    figurmh = figuremargin / figuresize[1]
    plotw = plotwidth / figuresize[0]
    ploth = plotwidth / figuresize[1]
    cbarmargin = cbarmargin_ / figuresize[0]
    cbarwidth = cbarwidth_ / figuresize[0]

    ax = []
    for n in range(rows):
        h = []
        for m in range(cols):
            if m == 0:
                h.append([ figurm, figurmh + ((rows - n - 1) * ploth), plotw, ploth])
            else:
                h.append([ h[m-1][0] + plotw , h[m-1][1], plotw, ploth ])
        ax.append(h)

    if colorbar:
        left, bottom = figurm + (cols * plotw) + cbarmargin, figurmh
        ax.append([[left, bottom, cbarwidth, rows * ploth]])

    return ax, figuresize

# axs, figs = makemapgrid(2, 2, colorbar=True)
# ax = []
# fig = pl.figure(figsize=figs)
# for h in axs:
#     for a in h:
#         fig.add_axes(a)
# pl.show()

def make_grid(rows, cols, scale = 1):
    #in inches
    figuremargin_ = 1 * scale
    plotwidth_ = 2.5 * scale
    plotsep_ = 1

    figuresize = ( (figuremargin_ * 2) + (cols * plotwidth_) + ((cols-1)*plotsep_), (figuremargin_ * 2) + (rows * plotwidth_) + ((rows-1)*plotsep_) )

    #in percents
    figurm = figuremargin_ / figuresize[0]
    figurmh = figuremargin_ / figuresize[1]
    plotw = plotwidth_ / figuresize[0]
    ploth = plotwidth_ / figuresize[1]
    plotsepw = plotsep_ / figuresize[0]
    plotseph = plotsep_ / figuresize[1]

    ax = []
    for n in range(cols):
        for m in range(rows):

            left = figurm + (n * (plotw + plotsepw))
            bottom = figurmh + ((rows-1) * (ploth + plotseph)) - (m * (ploth + plotseph))

            ax.append([ left, bottom, plotw, ploth ])

    return ax, figuresize

# axs, figs = make_grid(3, 3)
# ax = []
# fig = pl.figure(figsize=figs)
# for a in axs:
#     ax.append(fig.add_axes(a))
# pl.show()

def customPlot_topbar(width = 4, height = 4, edgemargin = 1):

    margin_ = edgemargin
    plotwidth_ = width
    plotheight_ = height
    sidebarwidth_ = .35
    sidebarmargin_ = .125
    insetheight_ = .25
    insetwidth_ = plotwidth_ / 4
    insetmargin_ = .125
    
    figuresize = ( plotwidth_ + (2 * margin_) , plotheight_ + sidebarwidth_+(2 * margin_) ) 

    marginh = margin_ / figuresize[1]
    marginw = margin_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]

    insetwidth = insetwidth_ / figuresize[0]
    insetheight = insetheight_ / figuresize[1]
    insetmarginw = insetmargin_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]

    sidebarwidth = sidebarwidth_ / figuresize[0]
    sidebarmargin = sidebarmargin_ / figuresize[0]

    ax = []


            #left, bottom, width, height
    
    left, bottom = marginw, marginh
    ax.append([left, bottom, plotwidth, plotheight])

    # left, bottom = marginw, marginh + plotheight-sidebarwidth
    # ax.append([left, bottom, plotwidth*.272727, sidebarwidth])

    # left, bottom = marginw + plotwidth*.727272727, marginh + plotheight-sidebarwidth
    # ax.append([left, bottom, plotwidth*.272727, sidebarwidth])

    # left, bottom = marginw + plotwidth*.272727, marginh + plotheight-sidebarwidth
    # ax.append([left, bottom, plotwidth*.45454, sidebarwidth])

    left, bottom = marginw, marginh + plotheight-sidebarwidth
    ax.append([left, bottom, plotwidth/3, sidebarwidth])

    left, bottom = marginw + plotwidth/3 + plotwidth/3, marginh + plotheight-sidebarwidth
    ax.append([left, bottom, plotwidth/3, sidebarwidth])

    left, bottom = marginw + plotwidth/3, marginh + plotheight-sidebarwidth
    ax.append([left, bottom, plotwidth/3, sidebarwidth])



    return ax, figuresize

def fig_1():
    totalwidth = 14
    edge_ = 0
    marginL_ = .2
    marginR_ = .2
    marginT_ = .2
    marginB_ = .7
    smallplotsep_ = .2
    plotsepw_ = .9
    plotseph_ = .7
    # rightplotwidth_ = (totalwidth - marginL_ - marginR_ - (plotsepw_ * 2))/3
    # rightplotheight_ = rightplotwidth_
    plotwidth_ = (totalwidth - marginL_ - marginR_ - (plotsepw_ * 2))/3
    plotheight_ = plotwidth_
    # plotwidth_ = (totalwidth - rightplotwidth_- marginL_ - marginR_ - smallplotsep_)/2
    # plotheight_ = rightplotheight_
    largewidth_ = plotwidth_ + edge_
    smallplotwidth_ = (plotwidth_ - plotsepw_)/2
    smallplotheight_ = smallplotwidth_
    insetplotheight_ = smallplotheight_ * .9
    insetplotwidth_ = smallplotwidth_ * .9
    insetmargin_ = .1

    # sidebarwidth_ = .2
    # sidebarmargin_ = .1
    # insetbarh_ = insetplotheight_ * .4
    # insetbarw_ = insetplotwidth_ * .06
    insetbarh_ = smallplotheight_ * .4
    insetbarw_ = smallplotwidth_ * .06
    maininsetbarh_ = plotheight_ * .4
    maininsetbarw_ = plotwidth_ * .06
    # insetmargin_ = .125
    # insetmarginw_ = .25

    figuresize = ( totalwidth, (plotheight_) + (marginB_ + marginT_) ) 

    marginL = marginL_ / figuresize[0]
    marginR = marginR_ / figuresize[0]
    marginT = marginT_ / figuresize[1]
    marginB = marginB_ / figuresize[1]
    plotsepw = plotsepw_ / figuresize[0]
    plotseph = plotseph_ / figuresize[1]
    smallplotsepw = smallplotsep_ / figuresize[0]
    boxwidth = largewidth_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    smallplotwidth = smallplotwidth_ / figuresize[0]
    smallplotheight = smallplotheight_ / figuresize[1]
    insetplotheight = insetplotheight_ / figuresize[1]
    insetplotwidth = insetplotwidth_ / figuresize[0]
    insetmarginw = insetmargin_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]

    insetbarh = insetbarh_ / figuresize[1]
    insetbarw = insetbarw_ / figuresize[0]
    maininsetbarh = maininsetbarh_ / figuresize[1]
    maininsetbarw = maininsetbarw_ / figuresize[0]

    edge = edge_ / figuresize[0]

    ax = []


    # #left
    # left, bottom = marginL-edge, marginB
    # ax.append([left, bottom, boxwidth, plotheight])

    # #middle
    # left, bottom = marginL + plotsepw + plotwidth - edge, marginB
    # ax.append([left, bottom, boxwidth, plotheight])

    #right
    left, bottom = marginL + plotsepw + plotwidth + plotsepw + plotwidth, marginB
    ax.append([left, bottom, plotwidth, plotheight])

    #left
    left, bottom = marginL, marginB
    ax.append([left, bottom, plotwidth, plotheight])

    #middle
    left, bottom = marginL + plotsepw + plotwidth, marginB
    ax.append([left, bottom, plotwidth, plotheight])



    return ax, figuresize

# axs, figs = fig_1()
# ax = []
# fig = pl.figure(figsize=figs)
# for a in axs:
#     ax.append(fig.add_axes(a))
# for i in range(len(ax)):

#     ax[i].set_ylabel("ABC abs wef", fontsize = 14, labelpad = 0)
#     ax[i].set_xlabel("ABC abs wef", fontsize = 14, labelpad = 0)
#     ax[i].tick_params(axis='x', labelsize = 12, direction = 'in', color = 'k')
#     ax[i].tick_params(axis='y', labelsize = 12, direction = 'in', color = 'k')

# pl.show()

def fig_2():
    totalwidth = 7
    marginL_ = .7
    marginR_ = .7
    marginT_ = .5
    marginB_ = .5
    plotsepw_ = .9
    plotseph_ = .7
    plotwidth_ = totalwidth - marginL_ - marginR_
    plotheight_ = plotwidth_
    smallplotwidth_ = (plotwidth_ - plotsepw_)/2
    smallplotheight_ = smallplotwidth_
    insetplotheight_ = smallplotheight_ * .9
    insetplotwidth_ = smallplotwidth_ * .9
    insetmargin_ = .1

    # sidebarwidth_ = .2
    # sidebarmargin_ = .1
    # insetbarh_ = insetplotheight_ * .4
    # insetbarw_ = insetplotwidth_ * .06
    insetbarh_ = smallplotheight_ * .4
    insetbarw_ = smallplotwidth_ * .06
    maininsetbarh_ = plotheight_ * .4
    maininsetbarw_ = plotwidth_ * .06
    # insetmargin_ = .125
    # insetmarginw_ = .25

    figuresize = ( totalwidth, (plotheight_ + smallplotheight_) + (marginB_ + marginT_) + plotseph_ ) 

    marginL = marginL_ / figuresize[0]
    marginR = marginR_ / figuresize[0]
    marginT = marginT_ / figuresize[1]
    marginB = marginB_ / figuresize[1]
    plotsepw = plotsepw_ / figuresize[0]
    plotseph = plotseph_ / figuresize[1]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    smallplotwidth = smallplotwidth_ / figuresize[0]
    smallplotheight = smallplotheight_ / figuresize[1]
    insetplotheight = insetplotheight_ / figuresize[1]
    insetplotwidth = insetplotwidth_ / figuresize[0]
    insetmarginw = insetmargin_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]

    insetbarh = insetbarh_ / figuresize[1]
    insetbarw = insetbarw_ / figuresize[0]
    maininsetbarh = maininsetbarh_ / figuresize[1]
    maininsetbarw = maininsetbarw_ / figuresize[0]

    ax = []

    #top  left
    left, bottom = marginL, marginB + plotheight + plotseph
    ax.append([left, bottom, smallplotwidth, smallplotheight])

    #top right
    left, bottom = marginL + smallplotwidth + plotsepw, marginB + plotheight + plotseph
    ax.append([left, bottom, smallplotwidth, smallplotheight])

    #main
    left, bottom = marginL, marginB
    ax.append([left, bottom, plotwidth, plotheight])

    # left, bottom = marginL + insetmarginw, marginB + plotheight - insetplotheight - insetmarginh
    # ax.append([left, bottom, insetplotwidth, insetplotheight])

    #top right colorbar
    left, bottom = marginL + smallplotwidth + plotsepw + smallplotwidth - insetbarw - insetmarginw, marginB + plotheight + plotseph + insetmarginh + insetmarginh
    ax.append([left, bottom, insetbarw, insetbarh])
    #main colorbar
    left, bottom = marginL + plotwidth - insetmarginw - maininsetbarw, marginB + insetmarginh + insetmarginh
    ax.append([left, bottom, maininsetbarw, maininsetbarh])

    return ax, figuresize

# axs, figs = fig_2()
# ax = []
# fig = pl.figure(figsize=figs)
# for a in axs:
#     ax.append(fig.add_axes(a))
# for i in range(len(ax)):
#     if i == 3:
#         ax[i].tick_params(axis='x', labelsize = 12, direction = 'in', color = 'k')
#         ax[i].tick_params(axis='y', labelsize = 12, direction = 'in', color = 'k')
#         # ax[i].set_xlabel("ABC abs wef", fontsize = 12, labelpad = 0)
#         ax[i].set_ylabel("ABC abs wef", fontsize = 12, labelpad = 0)
#         ax[i].set_xticks([])
#         # ax[i].yaxis.tick_right()
#         # ax[i].yaxis.set_label_position("right")
#     elif i == 4 or i == 5:
#         ax[i].tick_params(axis='x', labelsize = 12, direction = 'in', color = 'k')
#         ax[i].tick_params(axis='y', labelsize = 12, direction = 'in', color = 'k')
#         ax[i].set_xticks([])
#         ax[i].set_ylabel("ABC abs wef", fontsize = 12, labelpad = 0)
#         # ax[i].yaxis.tick_right()
#         # ax[i].yaxis.set_label_position("right")
#     else:
#         ax[i].tick_params(axis='x', labelsize = 12, direction = 'in', color = 'k')
#         ax[i].tick_params(axis='y', labelsize = 12, direction = 'in', color = 'k')
#         ax[i].set_xlabel("ABC abs wef", fontsize = 12, labelpad = 0)
#         ax[i].set_ylabel("ABC abs wef", fontsize = 12, labelpad = 0)

# pl.show()

def fig_3():
    totalwidth = 7
    marginL_ = .7
    marginR_ = .7
    marginT_ = .5
    marginB_ = .5
    plotsepw_ = .9
    plotseph_ = .7
    plotwidth_ = totalwidth - marginL_ - marginR_
    plotheight_ = plotwidth_
    smallplotwidth_ = (plotwidth_ - plotsepw_)/2
    smallplotheight_ = smallplotwidth_
    insetplotheight_ = smallplotheight_ * 1.2
    insetplotwidth_ = smallplotwidth_ * 1.2
    insetmargin_ = .1

    # sidebarwidth_ = .2
    # sidebarmargin_ = .1
    # insetbarh_ = insetplotheight_ * .4
    # insetbarw_ = insetplotwidth_ * .06
    insetbarh_ = smallplotheight_ * .4
    insetbarw_ = smallplotwidth_ * .06
    maininsetbarh_ = plotheight_ * .4
    maininsetbarw_ = plotwidth_ * .06
    # insetmargin_ = .125
    # insetmarginw_ = .25

    figuresize = ( totalwidth, (plotheight_ + smallplotheight_) + (marginB_ + marginT_) + plotseph_ ) 

    marginL = marginL_ / figuresize[0]
    marginR = marginR_ / figuresize[0]
    marginT = marginT_ / figuresize[1]
    marginB = marginB_ / figuresize[1]
    plotsepw = plotsepw_ / figuresize[0]
    plotseph = plotseph_ / figuresize[1]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    smallplotwidth = smallplotwidth_ / figuresize[0]
    smallplotheight = smallplotheight_ / figuresize[1]
    insetplotheight = insetplotheight_ / figuresize[1]
    insetplotwidth = insetplotwidth_ / figuresize[0]
    insetmarginw = insetmargin_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]

    insetbarh = insetbarh_ / figuresize[1]
    insetbarw = insetbarw_ / figuresize[0]
    maininsetbarh = maininsetbarh_ / figuresize[1]
    maininsetbarw = maininsetbarw_ / figuresize[0]

    ax = []

    #top left
    left, bottom = marginL, marginB + plotheight + plotseph
    ax.append([left, bottom, smallplotwidth, smallplotheight])

    #top right
    left, bottom = marginL + smallplotwidth + plotsepw, marginB + plotheight + plotseph
    ax.append([left, bottom, smallplotwidth, smallplotheight])

    #main
    left, bottom = marginL, marginB
    ax.append([left, bottom, plotwidth, plotheight])

    #inset to main
    left, bottom = marginL + insetmarginw + (.5*insetmarginw), marginB + plotheight - insetplotheight - insetmarginh
    ax.append([left, bottom, insetplotwidth, insetplotheight])

    #main colorbar
    left, bottom = marginL + plotwidth - insetmarginw - maininsetbarw, marginB + insetmarginh + insetmarginh + (2.5*insetmarginh)
    ax.append([left, bottom, maininsetbarw, maininsetbarh])

    #top right colorbar
    left, bottom = marginL + smallplotwidth + plotsepw + insetmarginw, marginB + plotheight + plotseph + insetmarginh
    ax.append([left, bottom, insetbarw, insetbarh])

    #inset colorbar
    left, bottom = marginL + insetmarginw + insetmarginw+ (.5*insetmarginw), marginB + plotheight - insetmarginh - insetmarginh - insetbarh
    ax.append([left, bottom, insetbarw, insetbarh])

    return ax, figuresize

def fig_3_stack():
    totalwidth = 7
    marginL_ = .6
    marginR_ = .2
    marginT_ = .5
    marginB_ = .5
    plotsepw_ = .9
    smallplotsep_ = .3
    plotsepsmall_ = .1
    plotseph_ = .7
    plotwidth_ = (totalwidth - marginL_ - marginR_ - plotsepsmall_)/2
    plotheight_ = plotwidth_
    smallplotheight_ = (plotheight_ - smallplotsep_)/2
    smallplotwidth_ = smallplotheight_

    smallmargin_ = ((plotwidth_ - smallplotwidth_)/2) - .2

    insetplotheight_ = smallplotheight_ * 1.2
    insetplotwidth_ = smallplotwidth_ * 1.2
    insetmargin_ = .1

    # sidebarwidth_ = .2
    # sidebarmargin_ = .1
    # insetbarh_ = insetplotheight_ * .4
    # insetbarw_ = insetplotwidth_ * .06
    insetbarh_ = smallplotheight_ * .4
    insetbarw_ = smallplotwidth_ * .06
    maininsetbarh_ = plotheight_ * .4
    maininsetbarw_ = plotwidth_ * .06
    # insetmargin_ = .125
    # insetmarginw_ = .25

    figuresize = ( totalwidth, (plotheight_ * 2) + (marginB_ + marginT_) + plotseph_ ) 

    marginL = marginL_ / figuresize[0]
    marginR = marginR_ / figuresize[0]
    marginT = marginT_ / figuresize[1]
    marginB = marginB_ / figuresize[1]
    plotsepw = plotsepw_ / figuresize[0]
    plotseph = plotseph_ / figuresize[1]
    smallplotsep = smallplotsep_ / figuresize[1]
    plotsepsmall = plotsepsmall_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    smallplotwidth = smallplotwidth_ / figuresize[0]
    smallplotheight = smallplotheight_ / figuresize[1]
    insetplotheight = insetplotheight_ / figuresize[1]
    insetplotwidth = insetplotwidth_ / figuresize[0]
    insetmarginw = insetmargin_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]
    smallmargin = smallmargin_ / figuresize[0]

    insetbarh = insetbarh_ / figuresize[1]
    insetbarw = insetbarw_ / figuresize[0]
    maininsetbarh = maininsetbarh_ / figuresize[1]
    maininsetbarw = maininsetbarw_ / figuresize[0]

    ax = []

    #top left
    left, bottom = marginL + smallmargin, marginB + plotheight + plotseph + smallplotheight + smallplotsep
    ax.append([left, bottom, smallplotwidth, smallplotheight])

    #top left bottom
    left, bottom = marginL + smallmargin, marginB + plotheight + plotseph
    ax.append([left, bottom, smallplotwidth, smallplotheight])

    #top left bottom colorbar
    left, bottom = marginL + insetmarginw + smallmargin, marginB + insetmarginh + plotheight + plotseph
    ax.append([left, bottom, insetbarw, insetbarh])

    #top right
    left, bottom = marginL + plotwidth + plotsepsmall, marginB + plotheight + plotseph
    ax.append([left, bottom, plotwidth, plotheight])

    #bottom left
    left, bottom = marginL , marginB
    ax.append([left, bottom, plotwidth, plotheight])

    #bottom left colorbar
    left, bottom = marginL  + insetmarginw, marginB + plotheight - maininsetbarh - (1*insetmarginh)
    ax.append([left, bottom, maininsetbarw, maininsetbarh])

    #bottom right
    left, bottom = marginL +  plotwidth + plotsepsmall, marginB
    ax.append([left, bottom, plotwidth, plotheight])

    #bottom right colorbar
    left, bottom = marginL + plotwidth + plotsepsmall + insetmarginw, marginB + plotheight - maininsetbarh - (1*insetmarginh)
    ax.append([left, bottom, maininsetbarw, maininsetbarh])

    return ax, figuresize

# axs, figs = fig_3_stack()
# ax = []
# fig = pl.figure(figsize=figs)
# for a in axs:
#     ax.append(fig.add_axes(a))
# pl.show()

def fig_3_wide():
    totalwidth = 14
    marginL_ = .9
    marginR_ = .5
    marginT_ = .5
    marginB_ = .6
    plotsepw_ = .9
    plotseph_ = .3
    plotsepsmall_ = .1

    plotwidth_ = 3.1
    plotheight_ = plotwidth_
    smallplotwidth_ = (plotwidth_ - plotseph_)/2
    smallplotheight_ = smallplotwidth_

    
    # smallplotwidth_ = (14 - (3*plotseph_) - marginT_ - marginB_)/7
    # smallplotheight_ = smallplotwidth_
    # plotwidth_ = smallplotwidth_ * 2 + plotseph_
    # plotheight_ = plotwidth_

    # smallplotwidth_ = (plotwidth_ - plotsepw_)/2
    # smallplotheight_ = smallplotwidth_
    insetplotheight_ = smallplotheight_ * 1.2
    insetplotwidth_ = smallplotwidth_ * 1.2
    insetmargin_ = .1

    # sidebarwidth_ = .2
    # sidebarmargin_ = .1
    # insetbarh_ = insetplotheight_ * .4
    # insetbarw_ = insetplotwidth_ * .06
    maininsetbarh_ = plotheight_ * .4
    maininsetbarw_ = plotwidth_ * .06
    insetbarh_ = smallplotheight_ * .4
    insetbarw_ = smallplotwidth_ * .08
    # insetbarw_ = plotwidth_ * .06
    # insetmargin_ = .125
    # insetmarginw_ = .25

    figuresize = ( (3 * plotwidth_) + (2 * plotsepw_) + plotsepsmall_ + marginL_ + marginR_ + smallplotwidth_ , (plotheight_ + marginT_ + marginB_ ) )

    print(figuresize[0])

    marginL = marginL_ / figuresize[0]
    marginR = marginR_ / figuresize[0]
    marginT = marginT_ / figuresize[1]
    marginB = marginB_ / figuresize[1]
    plotsepw = plotsepw_ / figuresize[0]
    plotseph = plotseph_ / figuresize[1]
    plotsepsmall = plotsepsmall_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    smallplotwidth = smallplotwidth_ / figuresize[0]
    smallplotheight = smallplotheight_ / figuresize[1]
    insetplotheight = insetplotheight_ / figuresize[1]
    insetplotwidth = insetplotwidth_ / figuresize[0]
    insetmarginw = insetmargin_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]

    insetbarh = insetbarh_ / figuresize[1]
    insetbarw = insetbarw_ / figuresize[0]
    maininsetbarh = maininsetbarh_ / figuresize[1]
    maininsetbarw = maininsetbarw_ / figuresize[0]

    ax = []

    #top left
    left, bottom = marginL, marginB + smallplotheight + plotseph
    ax.append([left, bottom, smallplotwidth, smallplotheight])

    #bottom left
    left, bottom = marginL, marginB
    ax.append([left, bottom, smallplotwidth, smallplotheight])

    #bottom left colorbar
    left, bottom = marginL + insetmarginw, marginB + insetmarginh
    ax.append([left, bottom, insetbarw, insetbarh])

    #left large
    left, bottom = marginL + smallplotwidth + plotsepw, marginB
    ax.append([left, bottom, plotwidth, plotheight])


    #middle large
    left, bottom = marginL + smallplotwidth + plotsepw + plotwidth + plotsepw, marginB
    ax.append([left, bottom, plotwidth, plotheight])

    #middle colorbar
    left, bottom = marginL + smallplotwidth + plotsepw + plotwidth + plotsepw + insetmarginw, marginB + plotheight - maininsetbarh - (1*insetmarginh)
    ax.append([left, bottom, maininsetbarw, maininsetbarh])

    #right large
    left, bottom = marginL + smallplotwidth + plotwidth + plotsepw + plotsepsmall + plotwidth + plotsepw, marginB
    ax.append([left, bottom, plotwidth, plotheight])

    #Right colorbar
    left, bottom = marginL + smallplotwidth + plotwidth + plotsepw + plotsepsmall + plotwidth + plotsepw + insetmarginw, marginB + plotheight - maininsetbarh - (1*insetmarginh)
    ax.append([left, bottom, maininsetbarw, maininsetbarh])

    return ax, figuresize

def fig_si_dark():
    totalwidth = 14
    marginL_ = .9
    marginR_ = .5
    marginT_ = .5
    marginB_ = .6
    plotsepw_ = .9
    plotseph_ = .3
    plotsepsmall_ = .1

    plotwidth_ = 3.1
    plotheight_ = plotwidth_
    smallplotwidth_ = (plotwidth_ - plotseph_)/2
    smallplotheight_ = smallplotwidth_

    
    # smallplotwidth_ = (14 - (3*plotseph_) - marginT_ - marginB_)/7
    # smallplotheight_ = smallplotwidth_
    # plotwidth_ = smallplotwidth_ * 2 + plotseph_
    # plotheight_ = plotwidth_

    # smallplotwidth_ = (plotwidth_ - plotsepw_)/2
    # smallplotheight_ = smallplotwidth_
    insetplotheight_ = smallplotheight_ * 1.2
    insetplotwidth_ = smallplotwidth_ * 1.2
    insetmargin_ = .1

    # sidebarwidth_ = .2
    # sidebarmargin_ = .1
    # insetbarh_ = insetplotheight_ * .4
    # insetbarw_ = insetplotwidth_ * .06
    maininsetbarh_ = plotheight_ * .4
    maininsetbarw_ = plotwidth_ * .06
    insetbarh_ = smallplotheight_ * .4
    insetbarw_ = smallplotwidth_ * .08
    # insetbarw_ = plotwidth_ * .06
    # insetmargin_ = .125
    # insetmarginw_ = .25

    figuresize = ( (4 * plotwidth_) + (3 * plotsepw_) + marginL_ + marginR_, (plotheight_ + marginT_ + marginB_ ) )

    print(figuresize[0])

    marginL = marginL_ / figuresize[0]
    marginR = marginR_ / figuresize[0]
    marginT = marginT_ / figuresize[1]
    marginB = marginB_ / figuresize[1]
    plotsepw = plotsepw_ / figuresize[0]
    plotseph = plotseph_ / figuresize[1]
    plotsepsmall = plotsepsmall_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    smallplotwidth = smallplotwidth_ / figuresize[0]
    smallplotheight = smallplotheight_ / figuresize[1]
    insetplotheight = insetplotheight_ / figuresize[1]
    insetplotwidth = insetplotwidth_ / figuresize[0]
    insetmarginw = insetmargin_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]

    insetbarh = insetbarh_ / figuresize[1]
    insetbarw = insetbarw_ / figuresize[0]
    maininsetbarh = maininsetbarh_ / figuresize[1]
    maininsetbarw = maininsetbarw_ / figuresize[0]

    ax = []

    #left
    left, bottom = marginL, marginB
    ax.append([left, bottom, plotwidth, plotheight])

    #bottom left colorbar
    left, bottom = marginL + insetmarginw, marginB + insetmarginh
    ax.append([left, bottom, maininsetbarw, maininsetbarh])

    #middle
    left, bottom = marginL + plotwidth + plotsepw, marginB
    ax.append([left, bottom, plotwidth, plotheight])

    #right
    left, bottom = marginL + plotwidth + plotsepw + plotwidth + plotsepw, marginB
    ax.append([left, bottom, plotwidth, plotheight])

    #far right
    left, bottom = marginL + plotwidth + plotsepw + plotwidth + plotsepw + plotwidth + plotsepw, marginB
    ax.append([left, bottom, plotwidth, plotheight])

    return ax, figuresize

# axs, figs = fig_si_dark()
# ax = []
# fig = pl.figure(figsize=figs)
# for a in axs:
#     ax.append(fig.add_axes(a))
# for i in range(len(ax)):
#     if i == 3:
#         ax[i].tick_params(axis='x', labelsize = 10, direction = 'in', color = 'k')
#         ax[i].tick_params(axis='y', labelsize = 10, direction = 'in', color = 'k')
#         ax[i].set_xlabel("ABC abs wef", fontsize = 10, labelpad = 0)
#         ax[i].set_ylabel("ABC abs wef", fontsize = 10, labelpad = 0)
#         ax[i].yaxis.tick_right()
#         ax[i].yaxis.set_label_position("right")
#     elif i == 4 or i == 5:
#         ax[i].tick_params(axis='x', labelsize = 10, direction = 'in', color = 'k')
#         ax[i].tick_params(axis='y', labelsize = 10, direction = 'in', color = 'k')
#         ax[i].set_xticks([])
#         ax[i].set_ylabel("ABC abs wef", fontsize = 10, labelpad = 0)
#         ax[i].yaxis.tick_right()
#         ax[i].yaxis.set_label_position("right")
#     else:
    # ax[i].tick_params(axis='x', labelsize = 12, direction = 'in', color = 'k')
    # ax[i].tick_params(axis='y', labelsize = 12, direction = 'in', color = 'k')
    # ax[i].set_xlabel("ABC abs wef", fontsize = 12, labelpad = 0)
    # ax[i].set_ylabel("ABC abs wef", fontsize = 12, labelpad = 0)
# pl.show()

def fig_4():
    totalwidth = 7
    marginL_ = .8
    marginR_ = .5
    marginT_ = .5
    marginB_ = .5
    plotsepw_ = .9
    plotseph_ = 0
    plotwidth_ = totalwidth - marginL_ - marginR_
    plotheight_ = plotwidth_
    smallplotwidth_ = (plotwidth_ - plotsepw_)/2
    smallplotheight_ = smallplotwidth_
    insetplotheight_ = plotheight_ * .4
    insetplotwidth_ = insetplotheight_
    insetmargin_ = .15
    insetmarginbar_ = .08

    # sidebarwidth_ = .2
    # sidebarmargin_ = .1
    # insetbarh_ = insetplotheight_ * .4
    # insetbarw_ = insetplotwidth_ * .06
    insetbarh_ = smallplotheight_ * .4
    insetbarw_ = smallplotwidth_ * .06
    maininsetbarh_ = plotheight_ * .06
    maininsetbarw_ = plotwidth_ * .35
    # insetmargin_ = .125
    # insetmarginw_ = .25

    figuresize = ( totalwidth, (plotheight_ + plotheight_) + (marginB_ + marginT_) + plotseph_ ) 

    marginL = marginL_ / figuresize[0]
    marginR = marginR_ / figuresize[0]
    marginT = marginT_ / figuresize[1]
    marginB = marginB_ / figuresize[1]
    plotsepw = plotsepw_ / figuresize[0]
    plotseph = plotseph_ / figuresize[1]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    smallplotwidth = smallplotwidth_ / figuresize[0]
    smallplotheight = smallplotheight_ / figuresize[1]
    insetplotheight = insetplotheight_ / figuresize[1]
    insetplotwidth = insetplotwidth_ / figuresize[0]
    insetmarginw = insetmargin_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]

    insetmarginbarh = insetmarginbar_ / figuresize[1]
    insetbarh = insetbarh_ / figuresize[1]
    insetbarw = insetbarw_ / figuresize[0]
    maininsetbarh = maininsetbarh_ / figuresize[1]
    maininsetbarw = maininsetbarw_ / figuresize[0]

    ax = []

    #top
    left, bottom = marginL, marginB + plotheight + plotseph
    ax.append([left, bottom, plotwidth, plotheight])

    #top inset
    left, bottom = marginL + insetmarginw, marginB + plotheight + plotheight + plotseph - insetplotheight - insetmarginh
    ax.append([left, bottom, insetplotwidth, insetplotheight])

    #bottom
    left, bottom = marginL, marginB
    ax.append([left, bottom, plotwidth, plotheight])

    #main colorbar
    left, bottom = marginL + plotwidth - (2*insetmarginw) - maininsetbarw, marginB + plotheight + plotseph + plotheight - insetmarginbarh - maininsetbarh
    ax.append([left, bottom, maininsetbarw, maininsetbarh])

    # left, bottom = marginL + insetmarginw, marginB + plotheight - insetplotheight - insetmarginh
    # ax.append([left, bottom, insetplotwidth, insetplotheight])

    # left, bottom = marginL + smallplotwidth + plotsepw + insetmarginw, marginB + plotheight + plotseph + insetmarginh
    # ax.append([left, bottom, insetbarw, insetbarh])

    # left, bottom = marginL + insetmarginw + insetmarginw, marginB + plotheight - insetmarginh - insetmarginh - insetbarh
    # ax.append([left, bottom, insetbarw, insetbarh])

    return ax, figuresize

# def fig_4():
#     totalwidth = 7
#     marginL_ = .8
#     marginR_ = .5
#     marginT_ = .5
#     marginB_ = .5
#     plotsepw_ = .9
#     plotseph_ = .2

#     plotwidth_ = totalwidth - marginL_ - marginR_
#     plotheight_ = plotwidth_

#     smallplotheight_ = plotheight_ / 4
#     smallplotwidth_ = smallplotheight_

#     smallsep_ = .2
#     plotheight2_ = plotheight_ - smallsep_ - smallplotheight_

#     insetplotheight_ = plotheight_ * .4
#     insetplotwidth_ = insetplotheight_
#     insetmargin_ = .15
#     insetmarginbar_ = .08

#     # sidebarwidth_ = .2
#     # sidebarmargin_ = .1
#     # insetbarh_ = insetplotheight_ * .4
#     # insetbarw_ = insetplotwidth_ * .06
#     insetbarh_ = smallplotheight_ * .4
#     insetbarw_ = smallplotwidth_ * .06
#     maininsetbarh_ = plotheight_ * .06
#     maininsetbarw_ = plotwidth_ * .35
#     # insetmargin_ = .125
#     # insetmarginw_ = .25

#     figuresize = ( totalwidth, (plotheight_ + plotheight2_) + (marginB_ + marginT_) + plotseph_ + smallplotheight_ + smallsep_ ) 

#     marginL = marginL_ / figuresize[0]
#     marginR = marginR_ / figuresize[0]
#     marginT = marginT_ / figuresize[1]
#     marginB = marginB_ / figuresize[1]
#     plotsepw = plotsepw_ / figuresize[0]
#     plotseph = plotseph_ / figuresize[1]
#     plotwidth = plotwidth_ / figuresize[0]
#     plotheight = plotheight_ / figuresize[1]
#     plotheight2 = plotheight2_ / figuresize[1]
#     smallplotwidth = smallplotwidth_ / figuresize[0]
#     smallplotheight = smallplotheight_ / figuresize[1]
#     insetplotheight = insetplotheight_ / figuresize[1]
#     insetplotwidth = insetplotwidth_ / figuresize[0]
#     insetmarginw = insetmargin_ / figuresize[0]
#     insetmarginh = insetmargin_ / figuresize[1]
#     smallseph = smallsep_ / figuresize[1]

#     insetmarginbarh = insetmarginbar_ / figuresize[1]
#     insetbarh = insetbarh_ / figuresize[1]
#     insetbarw = insetbarw_ / figuresize[0]
#     maininsetbarh = maininsetbarh_ / figuresize[1]
#     maininsetbarw = maininsetbarw_ / figuresize[0]
    

#     ax = []

#     #top
#     left, bottom = marginL, marginB + plotheight2 + plotseph + smallseph + smallplotheight
#     ax.append([left, bottom, plotwidth, plotheight])

#     #top inset
#     left, bottom = marginL + insetmarginw, marginB + plotheight + plotheight + plotseph - insetplotheight - insetmarginh
#     ax.append([left, bottom, insetplotwidth, insetplotheight])

#     #bottom
#     left, bottom = marginL, marginB
#     ax.append([left, bottom, plotwidth, plotheight2])

#     #middle squares
#     left, bottom = marginL, marginB + plotheight2 + smallseph
#     ax.append([left, bottom, smallplotwidth, smallplotheight])

#     #main colorbar
#     left, bottom = marginL + plotwidth - (2*insetmarginw) - maininsetbarw, marginB + plotheight + plotseph + plotheight - insetmarginbarh - maininsetbarh
#     ax.append([left, bottom, maininsetbarw, maininsetbarh])


#     return ax, figuresize

# axs, figs = fig_4()
# ax = []
# fig = pl.figure(figsize=figs)
# for a in axs:
#     ax.append(fig.add_axes(a))
# for i in range(len(ax)):
#     if i == 1:
#         ax[i].tick_params(axis='x', labelsize = 10, direction = 'in', color = 'k')
#         ax[i].tick_params(axis='y', labelsize = 10, direction = 'in', color = 'k')
#         ax[i].set_xlabel("ABC abs wef", fontsize = 10, labelpad = 0)
#         ax[i].set_ylabel("ABC abs wef", fontsize = 10, labelpad = 0)
#         ax[i].yaxis.tick_right()
#         ax[i].yaxis.set_label_position("right")
#     elif i == 4 or i == 5:
#         ax[i].tick_params(axis='x', labelsize = 10, direction = 'in', color = 'k')
#         ax[i].tick_params(axis='y', labelsize = 10, direction = 'in', color = 'k')
#         ax[i].set_xticks([])
#         ax[i].set_ylabel("ABC abs wef", fontsize = 10, labelpad = 0)
#         ax[i].yaxis.tick_right()
#         ax[i].yaxis.set_label_position("right")
#     else:
#         ax[i].tick_params(axis='x', labelsize = 12, direction = 'in', color = 'k')
#         ax[i].tick_params(axis='y', labelsize = 12, direction = 'in', color = 'k')
#         ax[i].set_xlabel("ABC abs wef", fontsize = 12, labelpad = 0)
#         ax[i].set_ylabel("ABC abs wef", fontsize = 12, labelpad = 0)

# pl.show()

def customPlot(width = 4, height = 4, edgemargin = 1, sidebar = False, insetbar = False):

    margin_ = edgemargin
    plotwidth_ = width
    plotheight_ = height
    sidebarwidth_ = (width / 4) * .25
    sidebarmargin_ = (width / 4) * .125
    insetheight_ = plotheight_ / 2.5
    insetwidth_ = plotwidth_ / 19
    insetmargin_ = .125
    insetmarginw_ = .1
    
    figuresize = ( plotwidth_  + sidebarmargin_ + sidebarwidth_ + (2 * margin_) , plotheight_ + (2 * margin_) ) 

    marginh = margin_ / figuresize[1]
    marginw = margin_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]

    insetwidth = insetwidth_ / figuresize[0]
    insetheight = insetheight_ / figuresize[1]
    insetmarginw = insetmarginw_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]

    sidebarwidth = sidebarwidth_ / figuresize[0]
    sidebarmargin = sidebarmargin_ / figuresize[0]

    ax = []


            #left, bottom, width, height
    
    left, bottom = marginw, marginh
    ax.append([left, bottom, plotwidth, plotheight])
    if sidebar:
        left = marginw + sidebarmargin + plotwidth
        ax.append([left, bottom, sidebarwidth, plotheight])
    if insetbar:
        left = marginw + insetmarginw
        bottom = marginh + (plotheight - insetheight - insetmarginh)
        ax.append([left, bottom, insetwidth, insetheight])

    return ax, figuresize

def customPlot_rightinset(width = 4, height = 4, sidebar = False, insetbar = False):

    margin_ = (width / 4) * 1
    plotwidth_ = width
    plotheight_ = height
    sidebarwidth_ = (width / 4) * .25
    sidebarmargin_ = (width / 4) * .125
    insetheight_ = plotheight_ / 2.5
    insetwidth_ = plotwidth_ / 19
    insetmargin_ = .125
    insetmarginw_ = .1
    
    figuresize = ( plotwidth_  + sidebarmargin_ + sidebarwidth_ + (2 * margin_) , plotheight_ + (1.3 * margin_) ) 

    marginh = margin_ / figuresize[1]
    marginw = margin_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]

    insetwidth = insetwidth_ / figuresize[0]
    insetheight = insetheight_ / figuresize[1]
    insetmarginw = insetmarginw_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]

    sidebarwidth = sidebarwidth_ / figuresize[0]
    sidebarmargin = sidebarmargin_ / figuresize[0]

    ax = []


            #left, bottom, width, height
    
    left, bottom = marginw, marginh
    ax.append([left, bottom, plotwidth, plotheight])
    if sidebar:
        left = marginw + sidebarmargin + plotwidth
        ax.append([left, bottom, sidebarwidth, plotheight])
    if insetbar:
        left = marginw + plotwidth - insetmarginw - insetwidth
        bottom = marginh + insetmarginh
        ax.append([left, bottom, insetwidth, insetheight])

    return ax, figuresize

# axs, figs = customPlot_rightinset(4, 4, insetbar = True)
# ax = []
# fig = pl.figure(figsize=figs)
# for a in axs:
#     ax.append(fig.add_axes(a))
# for i in range(len(ax)):
#     if i < 4:
#         ax[i].set_xlabel("xxxxxxx")
#         ax[i].set_ylabel("yyyyyyy")
# pl.show()

def customPlot_topbar(width = 4, height = 4, sidebar = False, insetbar = False):

    margin_ = (width / 4) * 1
    plotwidth_ = width
    plotheight_ = height
    sidebarwidth_ = (width / 4) * .25
    sidebarmargin_ = (width / 4) * .125
    insetheight_ = .25
    insetwidth_ = plotwidth_ / 3
    insetmargin_ = .125
    insetmarginw_ = .25
    
    figuresize = ( plotwidth_   + (1.5 * margin_) , plotheight_ + (2 * margin_) + sidebarmargin_ + sidebarwidth_ ) 

    marginh = margin_ / figuresize[1]
    marginw = margin_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]

    insetwidth = insetwidth_ / figuresize[0]
    insetheight = insetheight_ / figuresize[1]
    insetmarginw = insetmarginw_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]

    sidebarwidth = sidebarwidth_ / figuresize[0]
    sidebarmargin = sidebarmargin_ / figuresize[0]

    ax = []


            #left, bottom, width, height
    
    left, bottom = marginw, marginh
    ax.append([left, bottom, plotwidth, plotheight])
    if sidebar:
        bottom = marginh + plotheight + sidebarmargin
        ax.append([left, bottom, plotwidth, sidebarwidth])
    if insetbar:
        left = marginw + insetmarginw
        bottom = marginh + (plotheight - insetheight - insetmarginh)
        ax.append([left, bottom, insetwidth, insetheight])

    return ax, figuresize

#a custom plot with no edges
def customPlot_noedges(width = 4, height = 4, sidebar = False, insetbar = False):

    margin_ = 0
    plotwidth_ = width
    plotheight_ = height
    sidebarwidth_ = 0
    sidebarmargin_ = 0
    insetheight_ = .25
    insetwidth_ = plotwidth_ / 4
    insetmargin_ = .125
    insetmarginw_ = .25
    
    figuresize = ( plotwidth_  + sidebarmargin_ + sidebarwidth_ + (2 * margin_) , plotheight_ + (2 * margin_) ) 

    marginh = margin_ / figuresize[1]
    marginw = margin_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]

    insetwidth = insetwidth_ / figuresize[0]
    insetheight = insetheight_ / figuresize[1]
    insetmarginw = insetmarginw_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]

    sidebarwidth = sidebarwidth_ / figuresize[0]
    sidebarmargin = sidebarmargin_ / figuresize[0]

    ax = []


            #left, bottom, width, height
    
    left, bottom = marginw, marginh
    ax.append([left, bottom, plotwidth, plotheight])
    if sidebar:
        left = marginw + sidebarmargin + plotwidth
        ax.append([left, bottom, sidebarwidth, plotheight])
    if insetbar:
        left = marginw + insetmarginw
        bottom = marginh + (plotheight - insetheight - insetmarginh)
        ax.append([left, bottom, insetwidth, insetheight])

    return ax, figuresize

def photosyn_plot():

    margin_ = 1
    plotwidth_ = 4
    plotheight_ = 7
    sidebarwidth_ = .08
    subplotwidth_ = 4
    subplotheight_ = 3
    sidebarmargin_ = .125
    insetheight_ = .25
    insetwidth_ = plotwidth_ / 4
    insetmargin_ = .125
    
    figuresize = ( plotwidth_  + sidebarmargin_ + sidebarwidth_ + (2 * margin_) , plotheight_ + subplotheight_ + sidebarwidth_ + (2 * margin_) ) 

    marginh = margin_ / figuresize[1]
    marginw = margin_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    subplotwidth = subplotwidth_ / figuresize[0]
    subplotheight = subplotheight_ / figuresize[1]

    insetwidth = insetwidth_ / figuresize[0]
    insetheight = insetheight_ / figuresize[1]
    insetmarginw = insetmargin_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]

    sidebarwidth = sidebarwidth_ / figuresize[0]
    sidebarmargin = sidebarmargin_ / figuresize[0]

    ax = []


            #left, bottom, width, height
    
    left, bottom = marginw, marginh + subplotheight + sidebarwidth
    ax.append([left, bottom, plotwidth, plotheight])
    left, bottom = marginw, marginh
    ax.append([left, bottom, subplotwidth, subplotheight])

    return ax, figuresize

def customBottomtop(width = 4, height = 4, sidebar = False, insetbar = False):

    margin_ = 1
    plotwidth_ = width
    plotheight_ = height/2
    sidebarwidth_ = .25
    sidebarmargin_ = .125
    insetheight_ = .25
    insetwidth_ = plotwidth_ / 4
    insetmargin_ = .125
    
    figuresize = ( plotwidth_  + sidebarmargin_ + sidebarwidth_ + (2 * margin_) , (plotheight_ * 2) + (2 * margin_) ) 

    marginh = margin_ / figuresize[1]
    marginw = margin_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]

    insetwidth = insetwidth_ / figuresize[0]
    insetheight = insetheight_ / figuresize[1]
    insetmarginw = insetmargin_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]

    sidebarwidth = sidebarwidth_ / figuresize[0]
    sidebarmargin = sidebarmargin_ / figuresize[0]

    ax = []


            #left, bottom, width, height
    
    left, bottom = marginw, marginh + plotheight
    ax.append([left, bottom, plotwidth, plotheight])
    left, bottom = marginw, marginh
    ax.append([left, bottom, plotwidth, plotheight])
    if sidebar:
        left = marginw + sidebarmargin + plotwidth
        ax.append([left, bottom, sidebarwidth, plotheight*2])
    if insetbar:
        left = marginw + insetmarginw
        bottom = marginh + (plotheight - insetheight - insetmarginh)
        ax.append([left, bottom, insetwidth, insetheight])

    return ax, figuresize


def fig1(showplot = [1,1,1,1,1,1,1]):
    #figure 1 has 6 subplots and one inset on the top right subplot
    #space should be allocated for the plots whether it is utilized or not
    #default is only data plots are built, if a plots value is set to zero
    #the method will return 0 instead of an arrray

    rows = 2
    cols = 3

    margin_ = 1
    plotwidth_ = 4
    plotheight_ = 4
    plotsep_ = 1.25
    insetwidth_ = 1.5
    insetheight_ = 1.5
    insetmargin_ = .15
    textcord_ = [.6, .2]
    
    figuresize = ( (cols * plotwidth_) + ((cols - 1) *  plotsep_) + (2 * margin_) , (rows * plotheight_) + ((rows - 1) *  plotsep_) + (2 * margin_) ) 

    marginh = margin_ / figuresize[1]
    marginw = margin_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    plotsepw = plotsep_ / figuresize[0]
    plotseph = plotsep_ / figuresize[1]
    insetwidth = insetwidth_ / figuresize[0]
    insetheight = insetheight_ / figuresize[1]
    insetmarginw = insetmargin_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]
    textcord = [textcord_[0] / figuresize[0] , textcord_[1] / figuresize[1]]

    ax = []
    textlocation = []
    for i in range(cols):
        for p in range(rows):

            #left, bottom, width, height
            if showplot[i + (p*cols)] == 1:
                left = marginw + (i * (plotwidth + plotsepw))
                bottom = marginh + ((rows - 1 - p) * (plotheight + plotseph))
                ax.append([left, bottom, plotwidth, plotheight])
                textlocation.append([left - textcord[0], bottom + textcord[1] + plotheight])
            else:
                ax.append(0)
                textlocation.append(0)

    if showplot[-1] == 1:
        left = ax[4][0] + insetmarginw
        bottom = ax[4][1] + (plotheight - insetheight - insetmarginh)
        ax.append([left, bottom, insetwidth, insetheight])
    
    return ax, figuresize, textlocation

def fig4_v4_sidebar():
    #figure 4 shows the wavelength dependant photocurrent and pl side by side

    rows = 2
    cols = 3

    margin_ = 1
    # plotwidth_ = 2.21
    # plotheight_ = 2.21
    plotsep_ = 1.25

    largewidth_ = 6
    largeheight_ = 6
    sidewidth_ = 2.5
    sideheight_ = 6


    barwidth_ = .3
    sidemargin_ = 1.5
    barmargin_ = .07

    plotwidth_ = ( ( sidewidth_ + sidemargin_ + largewidth_ + sidemargin_ + barwidth_ ) - plotsep_ ) / 2
    plotheight_ = plotwidth_

    cbarwidth_ = .2
    cbarheight_ = plotheight_ / 3

    textcord_ = [.6, .2]
    
    figuresize = ( largewidth_ + sidewidth_  + sidemargin_ + (2 * margin_) + barwidth_ + (2 *  barmargin_), largeheight_ + (2 * margin_)  ) 

    marginh = margin_ / figuresize[1]
    marginw = margin_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    plotsepw = plotsep_ / figuresize[0]
    plotseph = plotsep_ / figuresize[1]
    largewidth = largewidth_ / figuresize[0]
    largeheight = largeheight_ / figuresize[1]
    sidewidth = sidewidth_ / figuresize[0]
    sideheight = sideheight_ / figuresize[1]
    sidemargin = sidemargin_ / figuresize[0]
    barwidth = barwidth_ / figuresize[0]
    cbarwidth = cbarwidth_ / figuresize[0]
    cbarheight = cbarheight_ / figuresize[1]
    barmargin = barmargin_ / figuresize[0]
    textcord = [textcord_[0] / figuresize[0] , textcord_[1] / figuresize[1]]

    ax = []
    axcb = []
    textlocation = []

    #left, bottom, width, height

    
    
    ax.append([ marginw, marginh, sidewidth, sideheight ])
    axcb.append( [ ax[0][0] + barmargin + sidewidth, marginh, barwidth , sideheight] )
    ax.append( [ marginw + sidewidth + sidemargin ,marginh, largewidth, largeheight ] )
    
    

    for i in range(len(ax)):
        if i == 1:
            textlocation.append([ax[i][0] - textcord[0], ax[i][1] + textcord[1] + largeheight])

    return ax, axcb, figuresize, textlocation

def fig4_v4():
    #figure 4 shows the wavelength dependant photocurrent and pl side by side

    rows = 2
    cols = 3

    margin_ = 1
    # plotwidth_ = 2.21
    # plotheight_ = 2.21
    plotsep_ = 1.25

    largewidth_ = 6
    largeheight_ = 6
    sidewidth_ = 2.5
    sideheight_ = 6


    barwidth_ = .3
    sidemargin_ = 1.5
    barmargin_ = .07

    plotwidth_ = ( ( sidewidth_ + sidemargin_ + largewidth_ + sidemargin_ + barwidth_ ) - plotsep_ ) / 2
    plotheight_ = plotwidth_

    cbarwidth_ = .2
    cbarheight_ = plotheight_ / 3

    textcord_ = [.6, .2]
    
    figuresize = ( largewidth_ + sidewidth_  + sidemargin_ + (2 * margin_), largeheight_ + (2 * margin_) + barwidth_ + (2 *  barmargin_) ) 

    marginh = margin_ / figuresize[1]
    marginw = margin_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    plotsepw = plotsep_ / figuresize[0]
    plotseph = plotsep_ / figuresize[1]
    largewidth = largewidth_ / figuresize[0]
    largeheight = largeheight_ / figuresize[1]
    sidewidth = sidewidth_ / figuresize[0]
    sideheight = sideheight_ / figuresize[1]
    sidemargin = sidemargin_ / figuresize[0]
    barwidth = barwidth_ / figuresize[0]
    cbarwidth = cbarwidth_ / figuresize[0]
    cbarheight = cbarheight_ / figuresize[1]
    barmargin = barmargin_ / figuresize[0]
    textcord = [textcord_[0] / figuresize[0] , textcord_[1] / figuresize[1]]

    ax = []
    axcb = []
    textlocation = []

    #left, bottom, width, height

    axcb.append( [ marginw, marginh + sideheight + barmargin, sidewidth, barwidth ] )
    
    ax.append([ marginw, marginh, sidewidth, sideheight ])
    ax.append( [ marginw + sidewidth + sidemargin ,marginh, largewidth, largeheight ] )
    

    for i in range(len(ax)):
        if i == 1:
            textlocation.append([ax[i][0] - textcord[0], ax[i][1] + textcord[1] + largeheight])

    return ax, axcb, figuresize, textlocation


def fig2():
    #Figure 2 has 7 axes but 6 subplots
    #Actually im going to build this with 6 axes and 5 subplots -- the symmettry might be nice

    rows = 2
    cols = 3

    margin_ = 1
    # plotwidth_ = 2.21
    # plotheight_ = 2.21
    plotsep_ = 1.25

    largewidth_ = 6
    largeheight_ = 6
    sidewidth_ = 2
    sideheight_ = 6


    barwidth_ = .3
    sidemargin_ = .15
    barmargin_ = .07

    plotwidth_ = ( ( sidewidth_ + sidemargin_ + largewidth_ + sidemargin_ + barwidth_ ) - plotsep_ ) / 2
    plotheight_ = plotwidth_

    cbarwidth_ = .2
    cbarheight_ = plotheight_ / 3

    textcord_ = [.6, .2]
    
    figuresize = ( (2 * plotwidth_) + (1 *  plotsep_) + (2 * margin_), (plotheight_ + plotheight_ + largeheight_) + (2 *  plotsep_) + (2 * margin_) ) 

    marginh = margin_ / figuresize[1]
    marginw = margin_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    plotsepw = plotsep_ / figuresize[0]
    plotseph = plotsep_ / figuresize[1]
    largewidth = largewidth_ / figuresize[0]
    largeheight = largeheight_ / figuresize[1]
    sidewidth = sidewidth_ / figuresize[0]
    sideheight = sideheight_ / figuresize[1]
    sidemargin = sidemargin_ / figuresize[0]
    barwidth = barwidth_ / figuresize[0]
    cbarwidth = cbarwidth_ / figuresize[0]
    cbarheight = cbarheight_ / figuresize[1]
    barmargin = barmargin_ / figuresize[0]
    textcord = [textcord_[0] / figuresize[0] , textcord_[1] / figuresize[1]]

    ax = []
    axcb = []
    textlocation = []

    #left, bottom, width, height

    ax.append( [ marginw, 1 - marginh - plotheight, plotwidth, plotheight ] )
    ax.append( [ marginw + plotwidth + plotsepw , 1 - marginh - plotheight, plotwidth, plotheight ] )
    
    ax.append( [ marginw, ax[0][1] - plotseph - largeheight, sidewidth, sideheight ] )
    ax.append( [ marginw + sidewidth + sidemargin, ax[2][1], largewidth, largeheight ] )

    ax.append( [ marginw, ax[3][1] - plotseph - plotheight, plotwidth, plotheight ] )
    ax.append( [ marginw + plotsepw + plotwidth, ax[3][1] - plotseph - plotheight, plotwidth, plotheight ] )

    axcb.append([ ax[0][0] + barmargin , ax[0][1] + cbarheight + cbarheight - barmargin, cbarwidth, cbarheight   ])
    axcb.append([ ax[1][0] + barmargin , ax[1][1] + cbarheight + cbarheight - barmargin, cbarwidth, cbarheight   ])
    axcb.append([ ax[3][0] + largewidth + sidemargin , ax[3][1], barwidth, largeheight   ])
    
    for i in range(len(ax)):
        if i == 2:
            textlocation.append([ax[i][0] - textcord[0], ax[i][1] + textcord[1] + largeheight])
        elif i == 3:
            textlocation.append(0)
        else:
            textlocation.append([ax[i][0] - textcord[0], ax[i][1] + textcord[1] + plotheight])

    return ax, axcb, figuresize, textlocation

# def fig2():
#     #figure 2 has 7 subplots

'''
figure 1 has 2 plots on the left a large middle plot, and then three plots on the right
'''
def figure1_v2_mapstack():

    rows = 2
    cols = 2

    margin_ = 1
    plotwidth_ = 4
    plotheight_ = 4

    smallplotwidth_ = 2
    smallplotheight_ = 2

    largeplotwidth_ = 6
    largeplotheight_ = 6

    plotsepw_ = 2.3
    plotseph_ = 1.5

    insetwidth_ = 1.5
    insetheight_ = 1.5
    insetmargin_ = .15

    splitmargin_ = .08

    cbarw_ = .2
    cbarh_ = plotheight_
    cbarmargin_ = .08

    textcord_ = [.6, .2]
    
    figuresize = ( plotwidth_ + (2 * margin_) ), (2 * plotheight_) + (plotseph_) + (2 * margin_) 

    marginh = margin_ / figuresize[1]
    marginw = margin_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    plotsepw = plotsepw_ / figuresize[0]
    plotseph = plotseph_ / figuresize[1]
    insetwidth = insetwidth_ / figuresize[0]
    insetheight = insetheight_ / figuresize[1]
    insetmarginw = insetmargin_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]
    splitmargin = splitmargin_ / figuresize[1]

    largeplotw = largeplotwidth_ / figuresize[0]
    largeploth = largeplotheight_ / figuresize[1]
    smallplotw = smallplotwidth_ / figuresize[0]
    smallploth = smallplotheight_ / figuresize[1]

    cbarw = cbarw_ / figuresize[0]
    cbarh = cbarh_ / figuresize[0]
    cbarmargin = cbarmargin_ / figuresize[0]

    textcord = [textcord_[0] / figuresize[0] , textcord_[1] / figuresize[1]]

    ax = []
    textlocation = []

    left, bottom = marginw, marginh
    ax.append([left, bottom, plotwidth, plotheight + plotheight + plotseph])
    textlocation.append([left - textcord[0], bottom + textcord[1] + plotheight + plotheight + plotseph])

    return ax, figuresize, textlocation

    '''
figure 1 has 2 plots on the left a large middle plot, and then three plots on the right -- this is just the plot portion on the end
'''
def figure1_v2():

    rows = 2
    cols = 2

    margin_ = 1
    plotwidth_ = 4
    plotheight_ = 4

    smallplotwidth_ = 2.2
    smallplotheight_ = 2.2

    largeplotwidth_ = 6
    largeplotheight_ = 6

    plotsepw_ = 1.6
    plotseph_ = 1.1

    insetwidth_ = 1.1
    insetheight_ = .15
    insetmargin_ = .1

    splitmargin_ = .08

    cbarw_ = .2
    cbarh_ = plotheight_
    cbarmargin_ = .08

    textcord_ = [.6, .2]
    
    figuresize = ( largeplotwidth_ + (2 * margin_)) , (largeplotheight_ + smallplotheight_) + (plotseph_) + (2 * margin_) 

    marginh = margin_ / figuresize[1]
    marginw = margin_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    plotsepw = plotsepw_ / figuresize[0]
    plotseph = plotseph_ / figuresize[1]
    insetwidth = insetwidth_ / figuresize[0]
    insetheight = insetheight_ / figuresize[1]
    insetmarginw = insetmargin_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]
    splitmargin = splitmargin_ / figuresize[1]

    largeplotw = largeplotwidth_ / figuresize[0]
    largeploth = largeplotheight_ / figuresize[1]
    smallplotw = smallplotwidth_ / figuresize[0]
    smallploth = smallplotheight_ / figuresize[1]

    cbarw = cbarw_ / figuresize[0]
    cbarh = cbarh_ / figuresize[0]
    cbarmargin = cbarmargin_ / figuresize[0]

    textcord = [textcord_[0] / figuresize[0] , textcord_[1] / figuresize[1]]

    ax = []
    axcb = []
    textlocation = []

    left, bottom = marginw, marginh + largeploth + plotseph
    ax.append([left, bottom, smallplotw, smallploth])
    textlocation.append([left - textcord[0], bottom + textcord[1] + smallploth])

    axcb.append([left + insetmarginw, bottom + smallploth - insetmarginh - insetheight, insetwidth, insetheight])

    left, bottom = marginw + smallplotw + plotsepw, marginh + largeploth + plotseph
    ax.append([left, bottom, smallplotw, smallploth])
    textlocation.append([left - textcord[0], bottom + textcord[1] + smallploth])

    left, bottom = marginw , marginh
    ax.append([left, bottom, largeplotw, largeploth])
    textlocation.append([left - textcord[0], bottom + textcord[1] + largeploth])

    return ax, axcb, figuresize, textlocation

'''
Side by side plots with three stacked panels, one panel on the right side
'''
def figure3_v3():

    rows = 2
    cols = 2

    margin_ = 1
    plotwidth_ = 4
    plotheight_ = 4
    smallplotheight_ = 2
    plotsepw_ = 2.3
    plotseph_ = 1.5
    insetwidth_ = 1.5
    insetheight_ = 1.5
    insetmargin_ = .15

    splitmargin_ = .08

    cbarw_ = .2
    cbarh_ = plotheight_
    cbarmargin_ = .08

    textcord_ = [.6, .2]
    
    figuresize = ( (3 * plotwidth_) + (2 *  plotsepw_) + (2 * margin_) , ( (2 * plotheight_) + splitmargin_) + (2 * margin_) ) 

    marginh = margin_ / figuresize[1]
    marginw = margin_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    smallploth = smallplotheight_ / figuresize[1]
    plotsepw = plotsepw_ / figuresize[0]
    plotseph = plotseph_ / figuresize[1]
    insetwidth = insetwidth_ / figuresize[0]
    insetheight = insetheight_ / figuresize[1]
    insetmarginw = insetmargin_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]
    splitmargin = splitmargin_ / figuresize[1]

    cbarw = cbarw_ / figuresize[0]
    cbarh = cbarh_ / figuresize[0]
    cbarmargin = cbarmargin_ / figuresize[0]

    textcord = [textcord_[0] / figuresize[0] , textcord_[1] / figuresize[1]]

    ax = []
    axcb = []
    textlocation = []

    for i in range(2):

        left = marginw + (i * (plotwidth + plotsepw))
        bottom = marginh + (plotheight + splitmargin )
        bottom1 = marginh + smallploth
        bottom2 = marginh

        ax.append([left, bottom, plotwidth, plotheight])
        ax.append([left, bottom1, plotwidth, smallploth])
        ax.append([left, bottom2, plotwidth, smallploth])
        axcb.append([left + plotwidth + cbarmargin, bottom, cbarw, plotheight])

        textlocation.append([left - textcord[0], bottom + textcord[1] + plotheight])

    left = marginw + (2 * (plotwidth + plotsepw))
    bottom = marginh + (plotheight + splitmargin )
    ax.append([left, bottom, plotwidth, plotheight])
    textlocation.append([left - textcord[0], bottom + textcord[1] + plotheight])

    return ax, axcb, figuresize, textlocation

'''
figure 3 is a 2 x 2 grid of 2-layer plots to show the correlation between alpha and NDC
'''
def figure3_v1():

    rows = 2
    cols = 2

    margin_ = 1
    plotwidth_ = 4
    plotheight_ = 2
    plotsepw_ = 2.3
    plotseph_ = 1.5
    insetwidth_ = 1.5
    insetheight_ = 1.5
    insetmargin_ = .15

    splitmargin_ = .08

    cbarw_ = .2
    cbarh_ = plotheight_
    cbarmargin_ = .08

    textcord_ = [.6, .2]
    
    figuresize = ( (cols * plotwidth_) + ((cols-1) *  plotsepw_) + (plotsepw_ * .5) + (2 * margin_) , (rows * ((2 * plotheight_) + splitmargin_)) + ((rows - 1) *  plotseph_) + (2 * margin_) ) 

    marginh = margin_ / figuresize[1]
    marginw = margin_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    plotsepw = plotsepw_ / figuresize[0]
    plotseph = plotseph_ / figuresize[1]
    insetwidth = insetwidth_ / figuresize[0]
    insetheight = insetheight_ / figuresize[1]
    insetmarginw = insetmargin_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]
    splitmargin = splitmargin_ / figuresize[1]

    cbarw = cbarw_ / figuresize[0]
    cbarh = cbarh_ / figuresize[0]
    cbarmargin = cbarmargin_ / figuresize[0]

    textcord = [textcord_[0] / figuresize[0] , textcord_[1] / figuresize[1]]

    ax = []
    ax1 = []
    textlocation = []

    for i in range(cols):
        for p in range(rows):

            left = marginw + (i * (plotwidth + plotsepw))
            bottom = marginh + ((rows - 1 - p) * (plotheight + splitmargin + plotheight + plotseph))
            ax1.append([left, bottom, plotwidth, plotheight])
            splitbottom = bottom + plotheight + splitmargin
            ax.append([left, splitbottom, plotwidth, plotheight])
            textlocation.append([left - textcord[0], splitbottom + textcord[1] + plotheight])

    return ax, ax1, figuresize, textlocation



'''
2 x 2 grid of plots. Every plot has a colorbar except for the lower right
'''
def figure1_v3():

    rows = 2
    cols = 2

    margin_ = 1
    plotwidth_ = 4
    plotheight_ = 4
    plotsepw_ = 2
    plotseph_ = 1.2
    insetwidth_ = 1.5
    insetheight_ = 1.5
    insetmargin_ = .15

    cbarw_ = .2
    cbarh_ = plotheight_ / 2
    cbarmargin_ = .08

    textcord_ = [.6, .2]
    
    figuresize = ( (cols * plotwidth_) + ((cols-1) *  plotsepw_) + (plotsepw_ * .5) + (2 * margin_) , (rows * plotheight_) + ((rows - 1) *  plotseph_) + (2 * margin_) ) 

    marginh = margin_ / figuresize[1]
    marginw = margin_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    plotsepw = plotsepw_ / figuresize[0]
    plotseph = plotseph_ / figuresize[1]
    insetwidth = insetwidth_ / figuresize[0]
    insetheight = insetheight_ / figuresize[1]
    insetmarginw = insetmargin_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]

    cbarw = cbarw_ / figuresize[0]
    cbarh = cbarh_ / figuresize[0]
    cbarmargin = cbarmargin_ / figuresize[0]

    textcord = [textcord_[0] / figuresize[0] , textcord_[1] / figuresize[1]]

    ax = []
    axcb = []
    textlocation = []
    axescount = 0
    for i in range(cols):
        for p in range(rows):

            left = marginw + (i * (plotwidth + plotsepw))
            bottom = marginh + ((rows - 1 - p) * (plotheight + plotseph))
            ax.append([left, bottom, plotwidth , plotheight])
            textlocation.append([left - textcord[0], bottom + textcord[1] + plotheight])
            if axescount == 3:
                left = ax[-1][0] + cbarmargin
                bottom = ax[-1][1] + (plotheight / 8) - cbarmargin
                axcb.append([left, bottom, cbarw, cbarh])
            axescount += 1


    return ax, axcb, figuresize, textlocation


'''
array of PL maps
'''
def PL_map(rows, cols):


    margin_ = 1
    plotwidth_ = 4
    plotheight_ = 3
    plotsepw_ = 1.5
    plotseph_ = .5

    cbarw_ = .2
    cbarh_ = plotheight_
    cbarmargin_ = .08

    figuresize = ( (margin_ * 2) + (cols * (plotwidth_ + cbarw_ + cbarmargin_)) + ((cols - 1)*plotsepw_) ,  (margin_ * 2) + (rows * plotheight_) + ((rows-1)*plotseph_)) 

    marginh = margin_ / figuresize[1]
    marginw = margin_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    plotsepw = plotsepw_ / figuresize[0]
    plotseph = plotseph_ / figuresize[1]

    cbarw = cbarw_ / figuresize[0]
    cbarh = cbarh_ / figuresize[0]
    cbarmargin = cbarmargin_ / figuresize[0]

    ax = []
    axcb = []

    for p in range(rows):
        for i in range(cols):

            left = marginw + (i * (plotwidth + plotsepw + cbarmargin + cbarw))
            bottom = marginh + ((rows - 1 - p) * (plotheight + plotseph))
            ax.append([left, bottom, plotwidth , plotheight])


            left = left + cbarmargin + plotwidth
            axcb.append([left, bottom, cbarw, plotheight/2])

    return ax, axcb, figuresize


'''
3 plots in a coloumn, the top and bottom plots have colorbars
'''
def figure2_v3():

    margin_ = 1
    plotwidth_ = 4
    plotheight_ = 4
    plotsepw_ = 2.3
    plotseph_ = 1.5
    insetwidth_ = 1.5
    insetheight_ = 1.5
    insetmargin_ = .15

    cbarw_ = .2
    cbarh_ = plotheight_
    cbarmargin_ = .08

    largeplotwidth_ = plotwidth_ + cbarmargin_ + cbarw_
    largeplotheight_ = largeplotwidth_


    textcord_ = [.6, .2]
    
    figuresize = (largeplotwidth_ + (2 * margin_) , (2 * plotheight_) + largeplotheight_ + (2 *  plotseph_) + (2 * margin_) )

    marginh = margin_ / figuresize[1]
    marginw = margin_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    plotsepw = plotsepw_ / figuresize[0]
    plotseph = plotseph_ / figuresize[1]
    insetwidth = insetwidth_ / figuresize[0]
    insetheight = insetheight_ / figuresize[1]
    insetmarginw = insetmargin_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]

    cbarw = cbarw_ / figuresize[0]
    cbarh = cbarh_ / figuresize[0]
    cbarmargin = cbarmargin_ / figuresize[0]

    largeplotw = largeplotwidth_ / figuresize[0]
    largeploth = largeplotheight_ / figuresize[1]

    textcord = [textcord_[0] / figuresize[0] , textcord_[1] / figuresize[1]]

    ax = []
    axcb = []
    textlocation = []
    axescount = 0

    left, bottom = marginw, marginh + largeploth + plotheight + (2*plotseph)
    ax.append([left, bottom, plotwidth, plotheight])
    textlocation.append([left - textcord[0], bottom + textcord[1] + plotheight])
    left, bottom = ax[-1][0] + plotwidth + cbarmargin, ax[-1][1]
    axcb.append([left, bottom, cbarw, plotheight])

    left, bottom = marginw, marginh + plotheight + plotseph
    ax.append([left, bottom, largeplotw, largeploth])
    textlocation.append([left - textcord[0], bottom + textcord[1] + largeploth])

    left, bottom = marginw, marginh
    ax.append([left, bottom, plotwidth, plotheight])
    textlocation.append([left - textcord[0], bottom + textcord[1] + plotheight])
    left, bottom = ax[-1][0] + plotwidth + cbarmargin, ax[-1][1]
    axcb.append([left, bottom, cbarw, plotheight])


    return ax, axcb, figuresize, textlocation

'''
figure 2 is a 2 x 2 grid of plots. Every plot has a colorbar except for the lower right
'''
def figure2_v1():

    rows = 2
    cols = 2

    margin_ = 1
    plotwidth_ = 4
    plotheight_ = 4
    plotsepw_ = 2.3
    plotseph_ = 1.5
    insetwidth_ = 1.5
    insetheight_ = 1.5
    insetmargin_ = .15

    cbarw_ = .2
    cbarh_ = plotheight_
    cbarmargin_ = .08

    textcord_ = [.6, .2]
    
    figuresize = ( (cols * plotwidth_) + ((cols-1) *  plotsepw_) + (plotsepw_ * .5) + (2 * margin_) , (rows * plotheight_) + ((rows - 1) *  plotseph_) + (2 * margin_) ) 

    marginh = margin_ / figuresize[1]
    marginw = margin_ / figuresize[0]
    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    plotsepw = plotsepw_ / figuresize[0]
    plotseph = plotseph_ / figuresize[1]
    insetwidth = insetwidth_ / figuresize[0]
    insetheight = insetheight_ / figuresize[1]
    insetmarginw = insetmargin_ / figuresize[0]
    insetmarginh = insetmargin_ / figuresize[1]

    cbarw = cbarw_ / figuresize[0]
    cbarh = cbarh_ / figuresize[0]
    cbarmargin = cbarmargin_ / figuresize[0]

    textcord = [textcord_[0] / figuresize[0] , textcord_[1] / figuresize[1]]

    ax = []
    axcb = []
    textlocation = []
    axescount = 0
    for i in range(cols):
        for p in range(rows):

            left = marginw + (i * (plotwidth + plotsepw))
            bottom = marginh + ((rows - 1 - p) * (plotheight + plotseph))
            ax.append([left, bottom, plotwidth, plotheight])
            textlocation.append([left - textcord[0], bottom + textcord[1] + plotheight])
            if axescount < 3:
                left = ax[-1][0] + plotwidth + cbarmargin
                bottom = ax[-1][1]
                axcb.append([left, bottom, cbarw, plotheight])
            axescount += 1


    return ax, axcb, figuresize, textlocation

'''
sets cbar params
'''
def triplecbarsettings(axcb):
    for a in axcb:
        if a is not axcb[0]:
            a.set_yticks([])
            # a.set_title("units", pad= 2)
            a.set_ylabel('').set_rotation(0)
            a.yaxis.set_label_position('right')
            a.yaxis.set_label_coords(1.15,.8)

    axcb[0].set_xticks([])
    axcb[0].yaxis.tick_right()
    axcb[0].yaxis.set_label_position('right')

'''
code repo
'''
def hold():

    axs, axcbs, figuresize = maketriple(3, 1)
    fig = pl.figure(figsize=figuresize)

    #################### AD ME ########################33
    ax = []
    axcb = []
    for a in axs:
        ax.append(fig.add_axes(a))
    for a in axcbs:
        axcb.append(fig.add_axes(a))
    #################### AD ME ########################33

    for a in axcb:
        if a is not axcb[0]:
            a.set_yticks([])
            # a.set_title("units", pad= 2)
            a.set_ylabel('units').set_rotation(0)
            a.yaxis.set_label_position('right')
            a.yaxis.set_label_coords(1.15,.8)
            
            # a.xaxis.set_label_position('top')
    axcb[0].set_xticks([])
    axcb[0].yaxis.tick_right()
    axcb[0].yaxis.set_label_position('right')
    axcb[0].set_title('units')

    for a in ax:
        a.set_ylabel('ylabel')
        a.set_xlabel('xlabel')
        a.set_title('subtitle')

    fig.suptitle("title")
    pl.show()