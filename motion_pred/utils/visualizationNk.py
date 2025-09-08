import os
import matplotlib.pyplot as plt
import numpy as np


def render_animationNK(skeleton, poses_generator, t_hist, fix_0=True, azim=0.0, output=None, size=6, ncol=6, scale=1):
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
    FPHAselect = {"1", "2", "3", "4", "5", "gt", "context"}
    HO3Dselect = {"1", "2", "3", "4", "5", "gt", "context"}
    HO3Dselect = {"1", "2", "3", "4", "5", "gt", "context"}

    HO3Dselect = {"gt", "context"}
    select = FPHAselect if scale == 3 else HO3Dselect
    poses = dict(filter(lambda x: x[0] in select, all_poses.items()))
    plt.ioff()
    nrow = 1  # int(np.ceil(len(poses) / ncol))  # ncol 12 nrow 1
    fig = plt.figure(figsize=(size * ncol, size * nrow))  # 6, 1
    ax_3d = []
    lines_3d = []
    radius = 3
    dataset = "FPHA" if scale == 3 else "HO3D"
    title = ["-15", "0", "15", "30", "45", "60"] if dataset == "FPHA" else ["-20", "0", "20", "40", "60", "80"]

    for index in range(6):  # start pose, gt motion, nk motion
        ax = fig.add_subplot(nrow, ncol, index + 1, projection="3d")
        ax.view_init(elev=15.0, azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_aspect("auto")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 6.0
        ax.set_title(title[index] + " sec", fontsize=40)
        ax.set_axis_off()
        ax.patch.set_alpha(0.5)
        ax_3d.append(ax)
        lines_3d.append([])

    fig.tight_layout()
    fig.subplots_adjust(wspace=-0.4, top=0.9)
    poses = list(poses.values())

    anim = None
    initialized = False

    his_Tcol, his_Icol, his_Mcol, his_Rcol, his_Lcol = "red", "black", "black", "black", "black"

    pred_Tcol, pred_Icol, pred_Mcol, pred_Rcol, pred_Lcol = "#AA2528", "#EA9C7F", "#DBDDE0", "#8AA4F0", "#4049B6"
    parents = skeleton.parents()

    def update_video(frame_index):
        FPHA_Frames = [0, 14, 29, 44, 59, 74]
        HO3D_Frames = [0, 19, 39, 59, 79, 99]

        frames = FPHA_Frames if dataset == "FPHA" else HO3D_Frames

        nonlocal initialized

        for n, ax in enumerate(ax_3d):
            if fix_0 and n == 0 and frame_index >= t_hist:
                continue
            for pose in poses:
                trajectories = pose[:, 0, [0, 1, 2]]  # wrist
                ax.set_xlim3d([-radius / 2 + trajectories[frames[n], 0], radius / 2 + trajectories[frames[n], 0]])
                ax.set_ylim3d([-radius / 2 + trajectories[frames[n], 1], radius / 2 + trajectories[frames[n], 1]])
                ax.set_zlim3d([-radius / 2 + trajectories[frames[n], 2], radius / 2 + trajectories[frames[n], 2]])

        if not initialized:
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                for n, ax in enumerate(ax_3d):

                    if n < 2:
                        Tcol, Icol, Mcol, Rcol, Lcol = his_Tcol, his_Icol, his_Mcol, his_Rcol, his_Lcol
                    else:
                        Tcol, Icol, Mcol, Rcol, Lcol = pred_Tcol, pred_Icol, pred_Mcol, pred_Rcol, pred_Lcol

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
                    for pose in poses:
                        pos = pose[frames[n]] * scale
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

                for n, ax in enumerate(ax_3d):
                    if fix_0 and n == 0 and frame_index >= t_hist:
                        continue

                    if n < 2:
                        Tcol, Icol, Mcol, Rcol, Lcol = his_Tcol, his_Icol, his_Mcol, his_Rcol, his_Lcol
                    else:
                        Tcol, Icol, Mcol, Rcol, Lcol = pred_Tcol, pred_Icol, pred_Mcol, pred_Rcol, pred_Lcol

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

                    for pose in poses:
                        pos = pose[frames[n]] * scale
                        lines_3d[n][j - 1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                        lines_3d[n][j - 1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
                        lines_3d[n][j - 1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir="z")
                        lines_3d[n][j - 1][0].set_color(col)

    def save():
        os.makedirs(output + "nk", exist_ok=True)
        nonlocal anim
        if anim is not None:
            anim.event_source.stop()

        update_video(0)
        plt.savefig(f"{output}/{action}_{fr_start}s_PromptFDDMv2.png")
        print(f"nk saved to {output}/PromptFDDMv2_{action}_{fr_start}s.png")

    save()
