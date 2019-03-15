#! /usr/bin/env python3

__all__ = [
    'CameraTrackRenderer'
]

from typing import List, Tuple

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders
from OpenGL import GLUT
from OpenGL.arrays import vbo

import data3d


def _build_cloud_program():
    example_vertex_shader = shaders.compileShader(
        """
        #version 130
        uniform mat4 mvp;

        in vec3 position;
        in vec3 vColor;
        
        out vec3 fColor;

        void main() {
            vec4 ndc_position = mvp * vec4(position, 1.0);
            gl_Position = ndc_position;
            
            fColor = vColor;
        }""",
        GL.GL_VERTEX_SHADER
    )
    example_fragment_shader = shaders.compileShader(
        """
        #version 130
        
        in vec3 fColor;
        
        out vec3 out_color;

        void main() {
            out_color = fColor;
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(
        example_vertex_shader, example_fragment_shader
    )


def _build_elements_program():
    example_vertex_shader = shaders.compileShader(
        """
        #version 130
        
        uniform mat4 mvp;

        in vec3 position;

        void main() {
            vec4 ndc_position = mvp * vec4(position, 1.0);
            gl_Position = ndc_position;
        }""",
        GL.GL_VERTEX_SHADER
    )
    example_fragment_shader = shaders.compileShader(
        """
        #version 130

        uniform vec3 color;

        out vec3 out_color;

        void main() {
            out_color = color;
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(
        example_vertex_shader, example_fragment_shader
    )


class CameraTrackRenderer:

    def __init__(self,
                 cam_model_files: Tuple[str, str],
                 tracked_cam_parameters: data3d.CameraParameters,
                 tracked_cam_track: List[data3d.Pose],
                 point_cloud: data3d.PointCloud):
        """
        Initialize CameraTrackRenderer. Load camera model, create buffer objects, load textures,
        compile shaders, e.t.c.

        :param cam_model_files: path to camera model obj file and texture. The model consists of
        triangles with per-point uv and normal attributes
        :param tracked_cam_parameters: tracked camera field of view and aspect ratio. To be used
        for building tracked camera frustrum
        :param point_cloud: colored point cloud
        """

        self._tracked_cam_track = tracked_cam_track
        self._tracked_cam_parameters = tracked_cam_parameters

        self._number_of_points = len(point_cloud.ids)

        points = point_cloud.points.reshape(-1).astype(np.float32)
        colors = point_cloud.colors.reshape(-1).astype(np.float32)

        self._points_buffer_object = vbo.VBO(points)
        self._colors_buffer_object = vbo.VBO(colors)

        self._cloud_program = _build_cloud_program()
        self._elements_program = _build_elements_program()

        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def display(self, camera_tr_vec, camera_rot_mat, camera_fov_y, tracked_cam_track_pos_float):
        """
        Draw everything with specified render camera position, projection parameters and 
        tracked camera position

        :param camera_tr_vec: vec3 position of render camera in global space
        :param camera_rot_mat: mat3 rotation matrix of render camera in global space
        :param camera_fov_y: render camera field of view. To be used for building a projection
        matrix. Use glutGet to calculate current aspect ratio
        :param tracked_cam_track_pos_float: a frame in which tracked camera
        model and frustrum should be drawn (see tracked_cam_track_pos for basic task)
        :return: returns nothing
        """

        # a frame in which a tracked camera model and frustrum should be drawn
        # without interpolation

        tracked_cam_track_pos = int(tracked_cam_track_pos_float)

        track = np.array(list(map(lambda pose: pose.t_vec, self._tracked_cam_track[:tracked_cam_track_pos + 1]))).astype(np.float32)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        model_matrix = np.eye(4, dtype=np.float32)
        model_matrix[1, 1] = -1.
        model_matrix[2, 2] = -1.

        view_matrix = self._calculate_view_matrix(camera_tr_vec, camera_rot_mat)
        projection_matrix = self._calculate_projection_matrix(camera_fov_y, 0.2, 100.)

        mvp = projection_matrix.dot(view_matrix.dot(model_matrix))

        self._render_cloud(mvp)
        self._render_camera_track(mvp, track)
        self._render_frustum(
            mvp,
            self.calculate_frustum_points(
                self._tracked_cam_track[tracked_cam_track_pos],
                self._tracked_cam_parameters
            ),
            self._tracked_cam_track[tracked_cam_track_pos].t_vec
        )

        GLUT.glutSwapBuffers()

    @staticmethod
    def _calculate_aspect_ratio():
        window_width = GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH)
        window_height = GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT)

        return window_width / window_height

    def _calculate_projection_matrix(self, fovy, znear, zfar):
        aspect_ratio = self._calculate_aspect_ratio()

        ymax = znear * np.tan(fovy / 2)
        xmax = ymax * aspect_ratio

        width = 2 * xmax
        height = 2 * ymax

        matrix = np.zeros((4, 4), dtype=np.float32)
        matrix[0, 0] = 2. * znear / width
        matrix[1, 1] = 2. * znear / height

        tmp = zfar - znear
        matrix[2, 2] = (-zfar - znear) / tmp
        matrix[2, 3] = (-2. * znear * zfar) / tmp

        matrix[3, 2] = -1.

        return matrix

    @staticmethod
    def _calculate_view_matrix(camera_tr_vec, camera_rot_mat):
        view_matrix = np.eye(4, dtype=np.float32)
        view_matrix[:, 3] = np.append(-camera_tr_vec, 1)

        rotation_inverse = np.linalg.inv(camera_rot_mat)
        rotation_inverse = np.vstack((rotation_inverse, [0., 0., 0.]))
        rotation_inverse = np.hstack((rotation_inverse, [[0.], [0.], [0.], [1.]]))
        return rotation_inverse.dot(view_matrix)

    def calculate_frustum_points(self, camera_pos: data3d.Pose, camera_parameters: data3d.CameraParameters):
        z = 10.

        ymax = z * np.tan(camera_parameters.fov_y)
        xmax = ymax * camera_parameters.aspect_ratio

        ur_corner = camera_pos.r_mat.dot(np.array([xmax, ymax, z])) + camera_pos.t_vec
        br_corner = camera_pos.r_mat.dot(np.array([xmax, -ymax, z])) + camera_pos.t_vec
        bl_corner = camera_pos.r_mat.dot(np.array([-xmax, -ymax, z])) + camera_pos.t_vec
        ul_corner = camera_pos.r_mat.dot(np.array([-xmax, ymax, z])) + camera_pos.t_vec

        return np.array([ur_corner, br_corner, bl_corner, ul_corner], dtype=np.float32)

    def _render_cloud(self, mvp):
        shaders.glUseProgram(self._cloud_program)

        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self._cloud_program, 'mvp'),
            1, True, mvp)

        self._points_buffer_object.bind()
        position_loc = GL.glGetAttribLocation(self._cloud_program, 'position')
        GL.glEnableVertexAttribArray(position_loc)
        GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                 False, 0,
                                 self._points_buffer_object)

        self._colors_buffer_object.bind()
        vColor_loc = GL.glGetAttribLocation(self._cloud_program, 'vColor')
        GL.glEnableVertexAttribArray(vColor_loc)
        GL.glVertexAttribPointer(vColor_loc, 3, GL.GL_FLOAT,
                                 False, 0,
                                 self._colors_buffer_object)

        GL.glDrawArrays(GL.GL_POINTS, 0, self._number_of_points)

        GL.glDisableVertexAttribArray(position_loc)
        GL.glDisableVertexAttribArray(vColor_loc)

        self._colors_buffer_object.unbind()
        self._points_buffer_object.unbind()

        shaders.glUseProgram(0)

    def _render_camera_track(self, mvp, track):
        self._render_elements(mvp, np.array([1.0, 1.0, 1.0], dtype=np.float32), track, GL.GL_LINE_STRIP)

    def _render_frustum(self, mvp, frustum_points, camera_pos):
        self._render_elements(mvp, np.array([1.0, 1.0, 0.0], dtype=np.float32), frustum_points, GL.GL_LINE_LOOP)

        connection_points = np.array(
            [camera_pos, frustum_points[0],
            camera_pos, frustum_points[1],
            camera_pos, frustum_points[2],
            camera_pos, frustum_points[3]],
            dtype=np.float32
        )

        self._render_elements(mvp, np.array([1.0, 1.0, 0.0], dtype=np.float32), connection_points, GL.GL_LINES)

    def _render_elements(self, mvp, color, points, mode):
        shaders.glUseProgram(self._elements_program)

        buffer = vbo.VBO(points.reshape(-1))

        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self._elements_program, 'mvp'),
            1, True, mvp)

        GL.glUniform3fv(
            GL.glGetUniformLocation(self._elements_program, 'color'),
            1, color
        )

        buffer.bind()
        position_loc = GL.glGetAttribLocation(self._elements_program, 'position')
        GL.glEnableVertexAttribArray(position_loc)
        GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                 False, 0,
                                 buffer)

        GL.glDrawArrays(mode, 0, points.shape[0])

        GL.glDisableVertexAttribArray(position_loc)

        buffer.unbind()

        shaders.glUseProgram(0)
