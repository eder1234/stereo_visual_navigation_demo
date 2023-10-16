import cv2
import os
import argparse

def video_to_frames(video_file, output_dir):
    """
    Convert a video file into a sequence of images.

    Parameters:
    - video_file: Path to the video file.
    - output_dir: Directory to save the extracted frames.

    Returns:
    - None
    """

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    vidcap = cv2.VideoCapture(video_file)

    # Check if video opened successfully
    if not vidcap.isOpened():
        print(f"Error: Cannot open video file {video_file}")
        return

    count = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break

        # Save the current frame as an image
        output_file = os.path.join(output_dir, f"frame_{count:04d}.jpg")
        cv2.imwrite(output_file, image)
        count += 1

    vidcap.release()
    print(f"Extracted {count} frames and saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a video to a sequence of images.')
    parser.add_argument('video_file', type=str, help='Path to the input video file.')
    parser.add_argument('output_dir', type=str, help='Path to the output directory for extracted frames.')

    args = parser.parse_args()

    video_to_frames(args.video_file, args.output_dir)

