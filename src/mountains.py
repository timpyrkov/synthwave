#!/usr/bin/env python
# -*- coding: utf8 -*-

import string
import itertools
import numpy as np
from scipy.stats import gennorm
from tqdm import tqdm
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mc
import colorsys
import cycler
import excolor


def perlin_to_grid(p):
    m, n = p.shape
    x = np.arange(m)
    y = np.arange(n)
    x, y = np.meshgrid(x, y, indexing="ij")
    idx = np.arange(x.size).reshape(x.shape)
    z = np.clip(p + 0.3, 0, None)
    return x, y, z, idx


def rotate(x, y):
    s = 1 / np.sqrt(2)
    x_ = s * x - s * y
    y_ = s * x + s * y
    return x_, y_


def perlin_expand_and_rotate(p):
    n = len(p)
    m = 3 * n + 1
    p = np.hstack([p] * 4)[:,:m]
    p = np.vstack([p] * 4)[:m]
    x, y, z, idx = perlin_to_grid(p)
    x = x / n - 1
    y = y / n - 1
    x, y = rotate(x, y)
    s = 1 / np.sqrt(2)
    s = gennorm.pdf(1.5 * (y - s), 8)
    z = z * s
    #z[z < 0.01] = 0.0
    # """ SANITY CHECK """
    # print(p.shape)
    # plt.figure(figsize=(32,24), facecolor="w")
    # plt.scatter(x, y, c=z, vmin=0)
    # plt.colorbar()
    # plt.grid(True)
    # plt.show()
    return x, y, z, idx

def grid_to_polygons(x, y, z, idx):
    f0 = np.stack([idx[:-1][:,:-1], idx[1:][:,:-1], idx[:-1][:,1:]]).reshape(3,-1)
    f1 = np.stack([idx[1:][:,:-1], idx[1:][:,1:], idx[:-1][:,1:]]).reshape(3,-1)
    faces = np.vstack([f0.T,f1.T])
    verts = np.stack([x.flatten(), y.flatten(), z.flatten()]).T
    # print(verts.shape, faces.shape)
    return verts, faces


# def perlin_to_polygons(p):
#     # m, n = p.shape
#     # x = np.arange(m)
#     # y = np.arange(n)
#     # x, y = np.meshgrid(x, y, indexing="ij")
#     x, y, z = perlin_to_grid(p)
#     idx = np.arange(x.size).reshape(x.shape)
#     f0 = np.stack([idx[:-1][:,:-1], idx[1:][:,:-1], idx[:-1][:,1:]]).reshape(3,-1)
#     f1 = np.stack([idx[1:][:,:-1], idx[1:][:,1:], idx[:-1][:,1:]]).reshape(3,-1)
#     faces = np.vstack([f0.T,f1.T])
#     verts = np.stack([x, y, z]).reshape(3,-1).T
#     # print(faces.shape, verts.shape)
#     return verts, faces


def face_normal(verts, faces):
    x = verts[faces[:,1]] - verts[faces[:,0]]
    y = verts[faces[:,2]] - verts[faces[:,0]]
    n = np.cross(x, y).T
    n /= np.linalg.norm(n, axis=0)
    n = n.T
    print(verts.shape, faces.shape)
    print(n.shape)
    return n


def face_color(norm, dark, light):
    """ TOP LIGHT """
    d = np.array([-1,-1,1])
    f = np.dot(norm, d)
    """ SIDE LIGHT """
    d = np.array([-1,-1,0])
    f = np.dot(norm, d) + 0.4
    # print("F", f.min(), f.max())
    f = np.clip(f, 0, 1)
    color1 = np.stack([mc.to_rgba(light)] * len(f)).T
    color2 = np.stack([mc.to_rgba(dark)] * len(f)).T
    color = f * color1 + (1 - f) * color2
    # print(color.shape, color)
    color = color.T
    return color



def plot_polygons(p, dark="navy", light="magenta", grid="cyan"):
    x, y, z, idx = perlin_expand_and_rotate(p)
    print("X.SHAPE", x.shape)
    verts, faces = grid_to_polygons(x, y, z, idx)
    norm = face_normal(verts, faces)
    color = face_color(norm, dark, light)
    fig = set_canvas(verts, grid=False)
    ax = plt.gca()
    for k, ind in enumerate(faces):
        srf = Poly3DCollection([verts[ind]], alpha=1, facecolor=color[k], edgecolor=grid, lw=1.5)
        ax.add_collection3d(srf)
    ax.view_init(elev=0, azim=90)
    ax.dist = 3
    img = excolor.image_to_array(fig)[::-1]
    plt.close()
    return img


def set_canvas(x, bg="#00000000", grid=False):
    fig = plt.figure(figsize=(48,48), facecolor=bg)
    ax = Axes3D(fig, auto_add_to_figure=False, proj_type="ortho")
    fig.add_axes(ax)
    x, y, z = x.T
    ax.set_xlim3d(x.min() + 0.5, x.max() - 0.5)
    ax.set_ylim3d(y.min() + 0.5, y.max() - 0.5)
    ax.set_zlim3d(z.min() - 1, z.max() + 1)
    ax.set_facecolor(bg)
    ax.set_xlabel("X label")
    ax.set_ylabel("Y label")
    ax.set_zlabel("Z label")
    if not grid:
        ax.w_xaxis.pane.fill = False
        ax.w_yaxis.pane.fill = False
        ax.w_zaxis.pane.fill = False
        ax.grid(False) 
        plt.axis("off")
    return fig



def shrink_y(img):
    n = img.shape[0]
    x = np.max(img[...,3], axis=1)
    idx = np.arange(n)[x > 0]
    i0, i1 = idx.min() - 5, idx.max() + 5
    newimg = img[i0:i1]
    # plt.figure(figsize=(18,6), facecolor="w")
    # plt.plot(x)
    # plt.show()
    return newimg


def shrink_x(img):
    n = img.shape[1] // 4
    diff = 255 * np.ones((n))
    for i in tqdm(range(1,n)):
        x0 = img[:,:i,:]
        x1 = img[:,-i:,:]
        diff[i] = np.mean(x1 - x0)
    # plt.figure(figsize=(18,6), facecolor="w")
    # plt.plot(diff)
    # plt.show()
    i = np.argmin(diff)
    newimg = img[:,i:,:]
    return newimg


def reshape_image(img):
    img = shrink_x(img)
    img = shrink_y(img)
    img = np.tile(img, (1,2,1))
    return img













