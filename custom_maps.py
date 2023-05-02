'''
custom plots -- mostly for figure and such
'''

#figure 4
def fig_4():
    #in inches
    plotwidth_ = 4
    plotheight_ = 4
    insetwidth_ = 1.6
    insetheight_ = 1.6
    barwidth_ = 1.4
    barheight_ = .22
    barleftinset_ = 2.35
    topinset_ = .05
    leftinset_ = .05
    figuremargin_ = 1.5


    #Calculates figuresize
    figuresize = ((plotwidth_) + (figuremargin_ * 2), (plotheight_ *2) + (figuremargin_ * 2))

    #converting to percents

    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    insetwidth = insetwidth_ / figuresize[0]
    insetheight = insetheight_ / figuresize[1]
    barwidth = barwidth_ / figuresize[0]
    barheight = barheight_ / figuresize[1]
    barleftinset = barleftinset_ / figuresize[0]
    topinset = topinset_ / figuresize[1]
    leftinset = leftinset_ / figuresize[0]
    figuremarginw = figuremargin_ / figuresize[0]
    figuremarginh = figuremargin_ / figuresize[1]

    ax = []
    #top plot
    left, bottom = figuremarginw, figuremarginh + plotheight
    ax.append([left, bottom, plotwidth, plotheight])
    #inset
    left, bottom = figuremarginw + leftinset, figuremarginh + plotheight + plotheight - insetheight - topinset
    ax.append([left, bottom, insetwidth, insetheight])
    #low plot
    left, bottom = figuremarginw, figuremarginh
    ax.append([left, bottom, plotwidth, plotheight])
    #cbar
    left, bottom = figuremarginw + barleftinset, figuremarginh + plotheight + plotheight - barheight - topinset
    ax.append([left, bottom, barwidth, barheight])

    return ax, figuresize


#SI diode: 2 rows 3 coloumn square
def SI_diode():
    #in inches
    plotwidth_ = 2
    plotheight_ = 2
    insetwidth_ = .8
    insetheight_ = .8
    barwidth_ = .7
    barheight_ = .11
    barleftinset_ = 2.35
    topinset_ = .05
    leftinset_ = .05
    figuremargin_ = 1.5
    plotsep_ = .9


    #Calculates figuresize
    figuresize = ((plotwidth_*3) + (figuremargin_ * 2) + (plotsep_ * 2), (plotheight_ *2) + plotsep_ + (figuremargin_ * 2))

    #converting to percents

    plotwidth = plotwidth_ / figuresize[0]
    plotheight = plotheight_ / figuresize[1]
    insetwidth = insetwidth_ / figuresize[0]
    insetheight = insetheight_ / figuresize[1]
    barwidth = barwidth_ / figuresize[0]
    barheight = barheight_ / figuresize[1]
    barleftinset = barleftinset_ / figuresize[0]
    topinset = topinset_ / figuresize[1]
    leftinset = leftinset_ / figuresize[0]
    figuremarginw = figuremargin_ / figuresize[0]
    figuremarginh = figuremargin_ / figuresize[1]
    plotsepw = plotsep_ / figuresize[0]
    plotseph = plotsep_ / figuresize[1]

    ax = []
    #top left
    left, bottom = figuremarginw, figuremarginh + plotheight + plotseph
    ax.append([left, bottom, plotwidth, plotheight])
    #top middle
    left, bottom = figuremarginw+plotsepw+plotwidth, figuremarginh + plotheight + plotseph
    ax.append([left, bottom, plotwidth, plotheight])
    #low left
    left, bottom = figuremarginw, figuremarginh
    ax.append([left, bottom, plotwidth, plotheight])
    #low middle
    left, bottom = figuremarginw+plotsepw+plotwidth, figuremarginh
    ax.append([left, bottom, plotwidth, plotheight])
    #top right
    left, bottom = figuremarginw+plotsepw+plotwidth+plotsepw+plotwidth, figuremarginh + plotheight + plotseph
    ax.append([left, bottom, plotwidth, plotheight])
    #low right
    left, bottom = figuremarginw+plotsepw+plotwidth+plotsepw+plotwidth, figuremarginh
    ax.append([left, bottom, plotwidth, plotheight])

    return ax, figuresize