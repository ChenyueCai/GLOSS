# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Portions adapted from CreativeFlow
# Source file: https://github.com/creativefloworg/creativeflow/blob/master/creativeflow/blender/geo_util.py
# Source licensed under MIT License
#
# Copyright (c) 2019 creativefloworg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Geometry/camera related utilities, mostly containing blender-specific code.
Heavily borrowed from: https://github.com/creativefloworg/creativeflow/blob/master/creativeflow/blender/geo_util.py
"""
import bpy
import bpy_extras

def get_single_camera_or_die():
    """
    Returns a single camera present in the scene (such as the one created
    by create_random_camera). Raises an exception if more than one or no camera
    found.

    Output:
    found camera
    """
    scene = bpy.context.scene
    camera = None
    for ob in scene.objects:
        if ob.type == 'CAMERA':
            if camera is not None:
                raise RuntimeError('More than one camera found')
            camera = ob
    if camera is None:
        raise RuntimeError('No camera found')
    return camera


def get_camera_by_number(camera_number):
    scene = bpy.context.scene

    cam_num = 0
    for ob in scene.objects:
        if ob.type == 'CAMERA':
            if cam_num == camera_number:
                return ob
            cam_num += 1
    return None


def save_blend(filename):
    """
    Saves current blend to file.

    Input:
    filename - string, absolute path to blend file
    """
    bpy.ops.wm.save_as_mainfile(filepath=filename)


def matrix_to_lists(matrix):
    return [list(r) for r in matrix]


def get_camera_information(cam):
    """
    Gets information for camera poses for all timeframes in the scene.
    """
    scene = bpy.context.scene
    start_frame = scene.frame_start
    end_frame = scene.frame_end

    # TODO: does this work?
    # Convert camera to the interpretable version
    assert cam.data.type == 'PERSP', f'Camera of type {cam.type} not supported; use perspective camera instead.'
    cam.data.lens_unit = 'FOV'

    # Note: there's lots more logic in the Blender NERF plug in, but not clear all of it makes sense;
    # e.g. angle_y does not seem to do anything for the camera in our scene.
    # Reference: https://github.com/maximeraafat/BlenderNeRF/blob/main/blender_nerf_operator.py
    output = {
        'camera_angle_x': cam.data.angle,  # Legacy
        'camera_intrinsics': {
            'width': scene.render.resolution_x,
            'height': scene.render.resolution_y,
            'fov': cam.data.angle,
            'shift_x': cam.data.shift_x,
            'shift_y': cam.data.shift_y,
            'near': cam.data.clip_start,
            'far': cam.data.clip_end
        }
    }
    frames = []
    for i in range(start_frame, end_frame + 1):
        scene.frame_set(i)

        frames.append({
            'view_matrix': matrix_to_lists(cam.matrix_world.inverted()),
            'transform_matrix': matrix_to_lists(cam.matrix_world),
            'id': i
        })
    output['frames'] = frames

    return output