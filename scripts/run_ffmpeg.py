#! /usr/bin/python3

import os, argparse
from os.path import join as join_path
from os.path import splitext as splitext

import shlex
from subprocess import Popen, PIPE

import multiprocessing
from multiprocessing import Pool, current_process
from multiprocessing.pool import ThreadPool


_ffmpeg_cmd = ('ffmpeg '
			  '-y '
			  '-i {input_video_path} '
			  '-vcodec libxvid '
			  '-s {width}x{height} '
			  '-r {frame_rate} '
			  '-an '
			  '-vcodec mpeg4 '
			  '{output_video_path}')


def convert_dataset_videos_constant_frame_rate(input_file_root, output_file_root, output_extension='.mp4', 
												height=224, width=224, frame_rate=30, n_jobs=20):
	""" Convert videos from a dataset root directory. 
	The folder structure of the video dataset must comply to the convension (uh... yeah the convension)

	Arguments
	------------------------------
		.. input_file_root
		.. output_file_root
		.. output_extension
		.. height
		.. width
		.. frame_rate
		.. n_jobs 
	"""
	# Search and get all input folders for classes
	input_class_dirs = abs_listdir(input_file_root)

	# Create list of output folders counterpart for each classes
	output_class_dirs = [join_path(output_file_root, folder) 
						for folder in os.listdir(input_file_root)]

	for input_class_folder, output_class_folder in zip(input_class_dirs, output_class_dirs):
		safe_mkdir(output_class_folder)

		# Search and get all video filenames in the input class folder
		input_video_paths = abs_listdir(input_class_folder)

		# Create list of output video name, this time we change the extension 
		# and save it to <output_file_root>/<output_class_folder>
		output_video_paths = [join_path(output_class_folder, splitext(f)[0] + output_extension)
						for f in os.listdir(input_class_folder)]

		print("- [run ffmpeg cfr] Current class folder: %s, total: %d" %(input_class_folder, len(input_video_paths)))
		print("- [run ffmpeg cfr] Output to folder: %s: " % output_class_folder)

		# Specify input video to convert and output video path
		run_args = list(zip(input_video_paths, output_video_paths))

		# Specify attributes of the output frames
		run_kwds = dict(height=height, width=width, frame_rate=frame_rate)

		# Create threads (max: your cpu count) and call ffmpeg program
		results = []
		pool = ThreadPool(min(n_jobs, multiprocessing.cpu_count()))

		for run_arg in run_args:
			result = pool.apply_async(_run_video_constant_frame_rate,
										args=run_arg,
										kwds=run_kwds)
			results.append(result)

		# Close the pool and wait for each task to complete
		pool.close()
		pool.join()

		for result in results:
			out, err = result.get()
			# Logging result
			with open("logs/ffmpeg_convert_cfr.txt", "a+") as f:
				f.write("PoolOutput: %s | Error: %s" %(out, err))


def _run_video_constant_frame_rate(input_video_path, output_video_path, height=224, width=224,
								  frame_rate=30):
	""" FFMPEG for video file input. Write video converted to constant framerate
	Arguments
	------------------------------
		.. input_video_path
		.. output_video_path
		.. height
		.. width
		.. frame_rate
	"""
	def call_proc(cmd):
		""" This runs in separate thread."""
		p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
		out, err = p.communicate()
		return out, err

	ffmpeg_cmd = _ffmpeg_cmd.format(input_video_path = input_video_path,
									output_video_path = output_video_path,
									width = width,
									height = height,
									frame_rate = frame_rate)

	print(".. Running ffmpeg convert constant frame rate on video %s" 
			% os.path.basename(input_video_path))

	return call_proc(ffmpeg_cmd)


def safe_mkdir(directory):
	try: 
		os.mkdir(directory)
	except (FileExistsError, OSError):
		pass  # folder has been created previously

	return os.path.abspath(directory)


def abs_listdir(directory):
	return [os.path.abspath(os.path.join(directory, f))
			for f in os.listdir(directory)]


if __name__ == '__main__':
	description = ('Parser for scripts/run_ffmpeg_cfr.py. ' 
					'This script convert videos to constant framerate, constant frame size/resolution \n'
					'and to other video format.' )
	parser = argparse.ArgumentParser(description=description)

	parser.add_argument('--input-file-root', type=str, required=True, help='Path to input video dir/root')
	parser.add_argument('--output-file-root', type=str, required=True, help='Path to output video dir/root')
	parser.add_argument('--output-extension', type=str, default='.mp4', help='Output video extension (default: .mp4)')
	parser.add_argument('--height', type=int, default=224, help='Output video frame height (default: 224)')
	parser.add_argument('--width', type=int, default=224, help='Output video frame width (default: 224)')
	parser.add_argument('--frame-rate', type=int, default=30, help='Output video constant frame rate (default: 30)')

	args = parser.parse_args()

	convert_dataset_videos_constant_frame_rate(input_file_root = args.input_file_root, 
												output_file_root = args.output_file_root,
												output_extension = args.output_extension,
												height = args.height,
												width = args.width,
												frame_rate = args.frame_rate)
