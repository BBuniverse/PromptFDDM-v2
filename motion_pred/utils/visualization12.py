# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import ipdb


def render_animation12(skeleton, poses_generator, t_hist, fix_0=True, azim=0.0, output=None, size=6, ncol=12, index_i=0, scale=1):
    """
    Render an animation. The supported output modes are:
    -- 'interactive': display an interactive figure
                    (also works on notebooks if associated with %matplotlib inline)
    -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
    -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
    -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    os.makedirs(output, exist_ok=True)
    all_poses = next(poses_generator)
    action = all_poses.pop("action")
    fr_start = all_poses.pop("fr_start")
    poses = all_poses
    plt.ioff()
    nrow = int(np.ceil(len(poses) / ncol))  # ncol 12 nrow 1
    fig = plt.figure(figsize=(size * ncol, size * nrow))
    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 3
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(nrow, ncol, index + 1, projection="3d")
        ax.view_init(elev=15.0, azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_aspect("auto")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 6
        ax.set_axis_off()
        ax.patch.set_alpha(0.5)
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    fig.tight_layout()
    fig.subplots_adjust(wspace=-0.4, top=0.9)
    poses = list(poses.values())

    anim = None
    initialized = False

    his_Tcol, his_Icol, his_Mcol, his_Rcol, his_Lcol = "red", "black", "black", "black", "black"
    pred_Tcol, pred_Icol, pred_Mcol, pred_Rcol, pred_Lcol = "#AA2528", "#EA9C7F", "#DBDDE0", "#8AA4F0", "#4049B6"

    parents = skeleton.parents()

    def update_video(i):
        nonlocal initialized

        if i < t_hist:
            Tcol, Icol, Mcol, Rcol, Lcol = his_Tcol, his_Icol, his_Mcol, his_Rcol, his_Lcol
        else:
            Tcol, Icol, Mcol, Rcol, Lcol = pred_Tcol, pred_Icol, pred_Mcol, pred_Rcol, pred_Lcol

        for n, ax in enumerate(ax_3d):
            if fix_0 and n == 0 and i >= t_hist:
                continue
            trajectories[n] = poses[n][:, 0, [0, 1, 2]]
            ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])
            ax.set_zlim3d([-radius / 2 + trajectories[n][i, 2], radius / 2 + trajectories[n][i, 2]])

        if not initialized:
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if j in skeleton.TFinger():
                    col = Tcol
                elif j in skeleton.IFinger():
                    col = Icol
                elif j in skeleton.MFinger():
                    col = Mcol
                elif j in skeleton.RFinger():
                    col = Rcol
                elif j in skeleton.LFinger():
                    col = Lcol

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i] * scale
                    lines_3d[n].append(
                        ax.plot(
                            [pos[j, 0], pos[j_parent, 0]],
                            [pos[j, 1], pos[j_parent, 1]],
                            [pos[j, 2], pos[j_parent, 2]],
                            zdir="z",
                            linewidth=5,
                            c=col,
                            marker="o",
                            markersize=8,
                            alpha=0.5,
                            aa=True,
                        )
                    )
            initialized = True
        else:
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if j in skeleton.TFinger():
                    col = Tcol
                elif j in skeleton.IFinger():
                    col = Icol
                elif j in skeleton.MFinger():
                    col = Mcol
                elif j in skeleton.RFinger():
                    col = Rcol
                elif j in skeleton.LFinger():
                    col = Lcol

                for n, ax in enumerate(ax_3d):
                    if fix_0 and n == 0 and i >= t_hist:
                        continue
                    pos = poses[n][i] * scale
                    lines_3d[n][j - 1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                    lines_3d[n][j - 1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
                    lines_3d[n][j - 1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir="z")
                    lines_3d[n][j - 1][0].set_color(col)

    def save():
        nonlocal anim
        os.makedirs(output + "end pose", exist_ok=True)
        if anim is not None:
            anim.event_source.stop()
        FPHA = [0, 74]
        HO3D = [0, 99]  # H2O
        frames = FPHA if scale == 3 else HO3D
        for i in frames:
            animation.FuncAnimation(fig, update_video(i), frames=1, interval=0, repeat=False)
            if i > 50:
                plt.savefig(f"{output}end pose/{action}_{fr_start}s_{i}_PromptFDDMv2.png")

    save()
