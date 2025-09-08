import os
from moviepy.editor import VideoFileClip, clips_array  # pip install moviepy==1.0.3
from moviepy.video.fx.all import crop


def create_stacked_gif_with_crop(video_paths, output_gif_path, fps=15):
    clips = []
    processed_clips = []
    final_clip = None

    try:
        clips = [VideoFileClip(path) for path in video_paths]

        # [[clip1], [clip2], [clip3]]
        final_clip = clips_array([[c] for c in clips])
        final_clip.write_gif(
            output_gif_path,
            fps=fps,
            opt="OptimizePlus",
            fuzz=50,
        )

        print(f"create GIF: {output_gif_path}")

    finally:
        for clip in clips + processed_clips:
            if clip:
                clip.close()
        if final_clip:
            final_clip.close()


if __name__ == "__main__":
    video_files = [
        "index/image/FPHA/Subject_4_pour milk 1_22s_GSPS.mp4",
        "index/image/FPHA/Subject_4_pour milk 1_22s_HumanMAC.mp4",
        "index/image/FPHA/Subject_4_pour milk 1_22s_PromptFDDMv2.mp4",
    ]

    video_files = [
        "index/image/HO3D/SMu1_mug 0_20s_GSPS.mp4",
        "index/image/HO3D/SMu1_mug 0_20s_HumanMAC.mp4",
        "index/image/HO3D/SMu1_mug 0_20s_PromptFDDMv2.mp4",
    ]

    video_files = [
        "index/image/H2O/subject4-k2-7-cam4-sub_1_359s_GSPS.mp4",
        "index/image/H2O/subject4-k2-7-cam4-sub_1_359s_HumanMAC.mp4",
        "index/image/H2O/subject4-k2-7-cam4-sub_1_359s_PromptFDDMv2.mp4",
    ]

    output_gif_file = "H2O_stacked_cropped_output.gif"

    final_gif_width = 6000
    final_gif_fps = 24

    create_stacked_gif_with_crop(
        video_paths=video_files,
        output_gif_path=output_gif_file,
        target_width=final_gif_width,
        fps=final_gif_fps,
    )
