#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:09:13 2020

@author: Marco
"""
def correlation_matrix(df3):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 50)
    cax = ax1.imshow(df3.corr(), cmap=cmap)
    plt.title('Correlation')
    labels=['Bilayer','IE','C$_{33}$','BG$_{PBE}$','BG$_{HSE}$','BG$_{min}$','BG$_{diff1}$','BG$_{diff2}$','BG$_{diff3}$',]
    labels2=['Bilayer','IE','C$_{33}$','BG$_{PBE}$','BG$_{HSE}$','BG$_{min}$','BG$_{diff1}$','BG$_{diff2}$','BG$_{diff3}$',]
    ax1.set_xticklabels(labels2,fontsize=12)
    ax1.set_yticklabels(labels,fontsize=12)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax)
    plt.show()

correlation_matrix(df3)
