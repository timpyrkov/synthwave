#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.stats import gennorm
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mc
import pylab as plt
from IPython.display import display, Image
from PIL import Image
import PIL
import excolor
import cv2
import gif



def draw_sun(size=(1080,1350), dpi=None, x=None, y=None, r=None, colors=["#FBD606", "#A917BE"]):
    """
    Generates image and mask for the Sun
    
    Parameters
    ----------
    size : tuple, default (1080,1350)
        Image width x height [pixels]
    dpi : int, optional
        Image resolution [dpi]
    x : float, optional
        X coordinate of the Sun center [pixels]
    y : float, optional
        Y coordinate of the Sun center [pixels]
    r : float, optional
        Radius of the Sun center [pixels]
    colors : array-like, optional
        Two colors (top, bottom)

    Returns
    -------
    img : ndarray
        3D array of pixels (nxy, nx, n_channel), channels = RGBA
    mask : ndarray
        2D array of pixels (nxy, nx)

    """
    nx, ny = size
    if dpi is None:
        figsize, dpi = excolor.size_to_size_and_dpi(size)
    else:
        figsize = (nx / dpi, ny / dpi)
    x = x if x is not None else nx // 2
    y = y if y is not None else 2 * ny // 3
    r = r if r is not None else ny // 5
    # Draw vertical color gradient
    a = 0.7 / r
    b = 0.7 * (r - y) / r
    f = a * np.arange(ny) + b
    f = np.clip(f, 0, 1).reshape(-1,1,1)
    f = np.repeat(f, nx, axis=1)
    f = np.repeat(f, 4, axis=2)
    img = f * mc.to_rgba(colors[0]) + (1 - f) * mc.to_rgba(colors[1])
    # Draq circle overlayed with horizontal bars
    nbars = 6
    fig = plt.figure(figsize=figsize, facecolor="w")
    fig.set_dpi(dpi)
    circle = plt.Circle((x, y ), r, color="k")
    plt.gca().add_patch(circle)
    dy = np.array([1 / (i + nbars // 2) for i in range(nbars)])
    dy = dy * r / np.sum(dy)
    for i in range(nbars):
        y0 = y - np.sum(dy[:i])
        bar = plt.Rectangle((0, y0), nx, -0.5 * dy[i], alpha=1, facecolor="w")
        plt.gca().add_patch(bar)
    excolor.remove_margins()
    plt.xlim(0, nx)
    plt.ylim(0, ny)
    plt.close()
    mask = excolor.image_to_array(fig, dpi=dpi)[::-1]
    mask = 1 - np.sum(mask[...,:3], axis=2) / 765
    img[...,-1] = mask
    # Draw transparency mask (0 - transparent, 1 - opaque)
    fig = plt.figure(figsize=figsize, facecolor="w")
    fig.set_dpi(dpi)
    circle = plt.Circle((x, y ), 1.0 * r, color="k")
    plt.gca().add_patch(circle)
    excolor.remove_margins()
    plt.xlim(0, nx)
    plt.ylim(0, ny)
    plt.close()
    mask = excolor.image_to_array(fig, dpi=dpi)[::-1]
    mask = np.sum(mask[...,:3], axis=2) / 765
    mask = cv2.GaussianBlur(mask, (r+1,r+1), r//2)
    return img, mask



def draw_moon(*args, **kwargs):
    """
    Generates image and mask for the Moon
    
    Parameters
    ----------
    size : tuple, default (1080,1350)
        Image width x height [pixels]
    dpi : int, optional
        Image resolution [dpi]
    x : float, optional
        X coordinate of the Moon center [pixels]
    y : float, optional
        Y coordinate of the Moon center [pixels]
    r : float, optional
        Radius of the Moon center [pixels]
    colors : array-like, optional
        Two colors (top, bottom)

    Returns
    -------
    img : ndarray
        3D array of pixels (nxy, nx, n_channel), channels = RGBA
    mask : ndarray
        2D array of pixels (nxy, nx)

    """
    kwargs["colors"] = ["#55D6F5", "#A917BE"]
    img, mask = draw_sun(*args, **kwargs)
    return img, mask


def background_fill(size=(1080,1350), colors="day"):
    """
    Generates background gradient and mask
    
    Parameters
    ----------
    size : tuple, default (1080,1350)
        Image width x height [pixels]
    colors : str or array-like, optional
        "day", "night" or list of colors (top to bottom)

    Returns
    -------
    img : ndarray
        3D array of pixels (nxy, nx, n_channel), channels = RGBA
    mask : ndarray
        2D array of pixels (nxy, nx)

    """
    nx, ny = size
    if not isinstance(colors, list) and not isinstance(colors, np.ndarray):
        if colors.lower() == "night":
            colors = ["#42007A", "#4F057A", "#5D097C"]
        else:
            colors = ["#550584", "#680B8E", "#A75466"]
    # Draw vertical color gradient
    rgb = np.array([mc.to_rgba(c) for c in colors])[::-1]
    rgb = rgb.reshape(-1,1,4)
    img = np.repeat(rgb, size[0], axis=1)
    idx = np.linspace(0,1,len(img))
    f = interp1d(idx, img, axis=0)
    idx = np.linspace(0,1, ny)
    img = f(idx)
    # Draw vertical transparency mask (0 -transparent, 1 - opaque)
    mask = np.linspace(1, -0.3, ny).reshape(-1,1)[::-1]
    mask = np.clip(mask, 0, 1)
    mask = np.repeat(mask, nx, axis=1)
    return img, mask


def generate_stars(nx, ny, seed=0):
    """
    Generates random stars using regular grid noise
    
    Parameters
    ----------
    nx : int
        Number of cells along the horizontal axis
    ny : int
        Number of cells along the vertical axis
    seed : int, optional
        Random seed

    Returns
    -------
    x : ndarray
        2D array of x-coordinates of stars
    y : ndarray
        2D array of y-coordinates of stars
    s : ndarray
        2D array of visible size of stars

    """
    x = np.arange(nx)
    y = np.arange(ny)
    x, y = np.meshgrid(x, y, indexing="ij")
    np.random.seed(seed)
    phi = np.random.uniform(0, 2 * np.pi, x.shape)
    r = 1e-3 / np.random.uniform(0, 0.5, x.shape)
    r = np.clip(0.5 - r, 0, None)
    z = r * np.exp(1j * phi)
    dx, dy = z.real, z.imag
    x = x + dx
    y = y + dy
    s = 10 * (0.5 - r) - 0.5
    s = np.clip(s, 0, None)
    return x, y, s


def draw_stars(size=(1080,1350), dpi=None, mask=None, seed=None):
    """
    Draws random stars using Cellular noise
    
    Parameters
    ----------
    size : tuple, default (1080,1350)
        Image size [pixels]
    dpi : int, optional
        Image resolution [dpi]
    mask : ndarray, optional
        2D array of transparency values (0 - transparent, 1 - opaque)
    seed : int, optional
        Random seed

    Returns
    -------
    img : ndarray
        3D array of pixels (nxy, nx, n_channel), channels = RGBA

    """
    if dpi is None:
        figsize, dpi = excolor.size_to_size_and_dpi(size)
    else:
        figsize = (size[0] / dpi, size[1] / dpi)
    nx, ny = 5 * int(figsize[0]), 5 * int(figsize[1])
    x, y, s = generate_stars(nx, ny, seed)
    fig = plt.figure(figsize=figsize, facecolor="#00000000")
    fig.set_dpi(dpi)
    plt.scatter(x, y, c="w", s=s, marker="d")
    plt.xlim(0,nx)
    plt.ylim(0,ny)
    excolor.remove_margins()
    img = excolor.image_to_array(fig, dpi=dpi)
    img[...,3] = (img[...,3] * mask).astype(np.uint8)
    plt.close()
    return img



def draw_ocean(i, p, size=(1080,1350), dpi=None, colors="day"):
    """
    Draws random waves using Perlin noise
    
    Parameters
    ----------
    i : int
        Time frame id - selects slice from 3D Perlin noise
    p : ndarray
        3D Perlin noise
    size : tuple, default (1080,1350)
        Image size [pixels]
    dpi : int, optional
        Image resolution [dpi]
    colors : str or array-like, optional
        "day", "night" or list of colors (dark to light)

    """
    nx, ny = size
    if dpi is None:
        figsize, dpi = excolor.size_to_size_and_dpi(size)
    else:
        figsize = (nx / dpi, ny / dpi)
    if not isinstance(colors, list) and not isinstance(colors, np.ndarray):
        if colors.lower() == "night":
            colors = ["#6260DC", "#55D6F5"]
        else:
            colors = ["#C45781", "#FDD806"]
    # Generate colors for the Moon/Sun track on the water
    n = nx // 10 + 1
    dn = (p[i].shape[-1] - n) // 2
    p_ = p[i][:,dn:n+dn]
    m, n = p_.shape
    wave_colors = []
    cmap = LinearSegmentedColormap.from_list("wave", colors)
    for j in range(m):
        sigma = (nx // 180) * (1 + 7 * j / m)
        gradient = signal.windows.general_gaussian(n, p=2, sig=sigma)
        wave_colors.append(cmap(gradient))
    # Draw waves
    dy = np.round((ny // 3) / p_.shape[0]).astype(int)
    dx = 1 if dpi > 200 else 0.6
    x = 10 * np.arange(n)
    for j in range(m):
        y = ny // 3 - dy * (j + 50 * p_[j])
        for k in range(len(y) - 1):
            x_ = [x[k], x[k+1] - dx]
            y_ = [y[k], y[k+1]]
            plt.plot(x_, y_, c=wave_colors[j][k], lw=0.4)
    return



def draw_ground(i=0, size=(1080,1350), dpi=None, color="#55D6F5"):
    """
    Draws ground grid
    
    Parameters
    ----------
    i : int, default 0
        Number of time frame
    size : tuple, default (1080,1350)
        Image size [pixels]
    dpi : int, optional
        Image resolution [dpi]
    color : str or matplotlib.colors.Color object, optional
        Grid color

    Returns
    -------
    img : ndarray
        3D array of pixels (nxy, nx, n_channel), channels = RGBA

    """
    nx, ny = size
    if dpi is None:
        figsize, dpi = excolor.size_to_size_and_dpi(size)
    else:
        figsize = (nx / dpi, ny / dpi)
    fig = plt.figure(figsize=figsize, facecolor="#00000000")
    fig.set_dpi(dpi)
    # Draw horizontal lines
    nbars = 10
    ymax = ny / 3
    dy = np.array([1 / (i + nbars // 2)**2 for i in range(nbars)])
    dy = dy * ymax / np.sum(dy)
    for k in range(nbars):
        y = np.sum(dy[:k+1])
        plt.plot([0, nx], [y, y], color=color, lw=3)
    # Draw radial perspective rays
    epsilon = 1e-5
    nstep = 40
    nrays = figsize[0]
    xmin, xmax = -nx, 2 * nx
    dx = (xmax - xmin) / nrays
    dt = np.modf(i / nstep)[0]
    for k in range(nrays):
        x0 = dx * (k + dt) - nx
        denominator = nx - 2 * x0
        if abs(denominator) > epsilon:
            a = ny / denominator
            b = -x0 * ny / denominator
            x1 = (ymax - b) / a
        else:
            x1 = x0
        plt.plot([x0, x1], [0, ymax], color=color, lw=3)
    plt.xlim(0, nx)
    plt.ylim(0, ny)
    excolor.remove_margins()
    plt.close()
    img = excolor.image_to_array(fig, dpi=dpi)[::-1]
    return img



def save_frame(fname, folder="~/Downloads/", dpi=180):
    """
    Decorator factory returns decorated function to
    plot and save one frame image to the "fname" folder
    
    Assume:
    1) args[0] should be integer frame index starting with 0
    2) func to be decoreted returns matplotlib.pyplot.Figure object

    Parameters
    ----------
    fname : str
        Stem part of desired "fname.gif"
    folder : str, default "~/Downloads/"
        Path to frames "folder/fname/"

    Returns
    -------
    decorator : function object
        Function to plot and save one frame imgage

    Example
    -------
    >>> from pyutils.animation import save_frame, make_gif
    >>> @save_frame('sample')
    >>> def plot_frame(i):
    >>>     fig = plt.figure()
    >>>     plt.scatter(0,0,s=100)
    >>>     return fig

    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            path = folder + "/" + fname + "/"
            path = os.path.expanduser(path)
            if not os.path.isdir(path):
                os.makedirs(path)
            if args[0] == 0:
                os.system(f"rm {path}/*.png")
            filename = f"{path}/f{args[0]:04d}.png"
            fig = func(*args, **kwargs)
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
            if isinstance(fig, PIL.Image.Image):
                fig.save(filename, dpi=dpi)
            else:
                plt.savefig(filename, dpi=dpi)
            plt.close()
            #cmd = f"convert {filename} -quality 20% {filename}"
            #os.system(cmd)
            return None
        return wrapper
    return decorator





















