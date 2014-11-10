import numpy as np
from vispy import gloo
from vispy.gloo import gl
from vispy import app
from vispy.util.transforms import perspective, translate, rotate
import transformations as tr
import sys
import random
import math

# vertex shader
vert = """
#version 120

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

attribute vec4 a_color;
attribute vec3  a_position;

varying vec4 v_color;

void main (void) {
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
    v_color = a_color;
    gl_PointSize = 1.0;
}
"""

# fragment shader
frag = """
#version 120

varying vec4 v_color;

void main()
{
	gl_FragColor = v_color;
}
"""

class Canvas(app.Canvas):

    def __init__(self, data_file):
        app.Canvas.__init__(self, keys='interactive')
        self.size = 800, 600
        self.title = 'Point Visualizer';

        if self.load_data(data_file) ==  False:
            print('Failed to load {0}'.format(data_file))
            sys.exit();

        # Model, View, Projection matrix
        self.view = np.eye(4, dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)
        #self.translate = 5
        self.translate = self.max_data_z*1.5
        translate(self.view, 0, 0, -self.translate)

        # Shader program for point data
        self.program_data = gloo.Program(vert, frag)
        self.program_data.bind(gloo.VertexBuffer(self.point_data))
        self.program_data['u_model'] = self.model
        self.program_data['u_view'] = self.view

        # Shader program for bbox or axis
        self.program_axis = gloo.Program(vert, frag)
        self.program_axis.bind(gloo.VertexBuffer(self.axis_data))
        self.program_axis['u_model'] = self.model
        self.program_axis['u_view'] = self.view

        # Shader program for plane
        self.program_plane = gloo.Program(vert, frag)
        self.program_plane.bind(gloo.VertexBuffer(self.plane_data))
        self.program_plane['u_model'] = self.model
        self.program_plane['u_view'] = self.view

        self.rot = [0,0,0,1]
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)


    def load_data(self, data_file):
        """ Load data from file and calculate bounding box
            Data format:
            x1 y1 z1
            x2 y2 z2
            ...
        """
        try:
            fr = open(data_file)
        except:
            return False

        point_list = []
        bbox = np.array([float('Inf'),-float('Inf'),float('Inf'),-float('Inf'),float('Inf'),-float('Inf')])
        for line in fr.readlines():
            xyz = [float(w) for w in line.split(' ')]
            point_list.append(xyz)
        fr.close();

        points = np.array(point_list)


        for xyz in points:
            bbox[0] = min(bbox[0], xyz[0]) # min x
            bbox[1] = max(bbox[1], xyz[0]) # max y
            bbox[2] = min(bbox[2], xyz[1]) # min x
            bbox[3] = max(bbox[3], xyz[1]) # max y
            bbox[4] = min(bbox[4], xyz[2]) # min z
            bbox[5] = max(bbox[5], xyz[2]) # max z

        bbox_corners = np.array([
            [bbox[0],bbox[2], bbox[4]],
            [bbox[0],bbox[2], bbox[5]],
            [bbox[0],bbox[3], bbox[5]],
            [bbox[0],bbox[3], bbox[4]],
            [bbox[1],bbox[3], bbox[4]],
            [bbox[1],bbox[2], bbox[4]],
            [bbox[1],bbox[2], bbox[5]],
            [bbox[1],bbox[3], bbox[5]]]);

        bbox_center = np.array([(bbox[0]+bbox[1])/2, (bbox[2]+bbox[3])/2, (bbox[4]+bbox[5])/2]);

        # RANSAC implementation
        # http://www.cse.yorku.ca/~kosta/CompVis_Notes/ransac.pdf
        # tolerance for distance, e.g. 0.0027m for kinect
        TOLERANCE = 0.0027
        # ratio of inliers
        THRESHOLD = 0.50
        N_ITERATIONS = 100
        iterations = 0
        solved = 0
        while iterations < N_ITERATIONS and solved == 0:
            iterations += 1
            max_error = -float('inf')
            max_index = -1
            # randomly pick three non-colinear points
            CP = np.array([0,0,0]);
            while CP[0] == 0 and CP[1] == 0 and CP[2] == 0:
                [A,B,C] = points[random.sample(xrange(len(points)), 3)];
                # make sure they are non-collinear
                CP = np.cross(A-B, B-C);
            # calculate plane coefficients
            abc = np.dot(np.linalg.inv(np.array([A,B,C])), np.ones([3,1]))
            # get distances from the plane
            d = math.sqrt(abc[0]*abc[0]+abc[1]*abc[1]+abc[2]*abc[2])
            dist = abs((np.dot(points, abc) - 1)/d)
            #print max(dist),min(dist)
            ind = np.where(dist < TOLERANCE)[0];
            ratio = float(len(ind))/len(points)
            if ratio > THRESHOLD:
                # satisfied, now fit model with the inliers
                # least squares reference plane: ax+by+cz=1
                inliers = np.take(points, ind, 0)
                print('\niterations: {0}, ratio: {1}, {2}/{3}'.format(iterations, ratio,len(points),len(inliers)))
                [a,b,c] = np.dot(np.linalg.pinv(inliers), np.ones([len(inliers), 1]))
                plane_pts = np.array([
                    [bbox[0], bbox[2], (1-a*bbox[0]-b*bbox[2])/c],
                    [bbox[0], bbox[3], (1-a*bbox[0]-b*bbox[3])/c],
                    [bbox[1], bbox[3], (1-a*bbox[1]-b*bbox[3])/c],
                    [bbox[1], bbox[2], (1-a*bbox[1]-b*bbox[2])/c]]);
                print('Least squares solution coeffiecients for ax+by+cz=1')
                print a,b,c;
                solved = 1

        if solved == 0:
            print('Can not find a good solution. Better luck next time...\n');
            sys.exit(0)


        self.point_data = np.zeros(len(points), [('a_position', np.float32, 3),('a_color', np.float32, 4)])
        self.point_data['a_position'] = points - bbox_center;
        self.point_data['a_color'] = [0,0,1,1]

        self.axis_data = np.zeros(8, [("a_position", np.float32, 3), ("a_color",    np.float32, 4)])
        self.axis_data['a_position'] = bbox_corners - bbox_center;
        self.axis_indies = gloo.IndexBuffer([0,1, 1,2, 2,3, 3,0, 4,7, 7,6, 6,5, 5,4, 0,5, 1,6, 2,7, 3,4 ])
        self.axis_data['a_color'] = [0,1,0,1]

        self.plane_data = np.zeros(4, [("a_position", np.float32, 3), ("a_color",    np.float32, 4)])
        self.plane_data['a_position'] = plane_pts - bbox_center;
        self.plane_indies = gloo.IndexBuffer([0,1,3, 1,2,3])
        self.plane_data['a_color'] = [0.5,0.8,1,0.75]


        self.max_data_z = bbox[5];

        return True

    def on_initialize(self, event):
		gloo.set_state('translucent', clear_color='white')

    def on_key_press(self, event):
        if event.text == ' ':
            if self.timer.running:
                self.timer.stop()
            else:
                self.timer.start()
        if event.text == 'f' or event.text == 'F':
            self.fullscreen = not self.fullscreen

    def on_timer(self, event):
        # auto rotating
        qy = tr.quaternion_about_axis(0.005, [0,0,1])
        qx = tr.quaternion_about_axis(0.005, [0,1,0])
        q = tr.quaternion_multiply(qx,qy)
        self.rot = tr.quaternion_multiply(self.rot, q)
        self.model = tr.quaternion_matrix(self.rot)
        self.program_data['u_model'] = self.model
        self.program_axis['u_model'] = self.model
        self.program_plane['u_model'] = self.model
        self.update()

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)
        self.projection = perspective(45.0, width / float(height), 1.0, 1000.0)
        self.program_data['u_projection'] = self.projection
        self.program_axis['u_projection'] = self.projection
        self.program_plane['u_projection'] = self.projection
        self.update()

    def on_mouse_wheel(self, event):
        # zoom in/out
        self.translate += event.delta[1]
        self.translate = max(2, self.translate)
        self.view = np.eye(4, dtype=np.float32)
        translate(self.view, 0, 0, -self.translate)

        self.program_data['u_view'] = self.view
        self.program_axis['u_view'] = self.view
        self.program_plane['u_view'] = self.view
        self.update()

    def on_mouse_press(self, event):
        self.last_drag_pos = event.pos;

    def on_mouse_move(self, event):
        # drag to rotate
        if event.is_dragging:# and not self.timer.running:
            # drag delta
            dx = event.pos[0]-self.last_drag_pos[0]
            dy = event.pos[1]-self.last_drag_pos[1]
            dampen = 0.05
            # quaternion rotation
            qy = tr.quaternion_about_axis(dx*dampen, [0,-1,0])
            qx = tr.quaternion_about_axis(dy*dampen, [-1,0,0])
            q = tr.quaternion_multiply(qx,qy)
            self.rot = tr.quaternion_multiply(self.rot, q)
            # rotate model
            self.model = tr.quaternion_matrix(self.rot)
            # update model
            self.program_data['u_model'] = self.model
            self.program_axis['u_model'] = self.model
            self.program_plane['u_model'] = self.model
            self.update()

            self.last_drag_pos = event.pos
        pass

    def on_draw(self, event):
        gloo.clear()
        self.program_data.draw(gl.GL_POINTS)
        self.program_axis.draw(mode = gl.GL_LINES, indices = self.axis_indies)
        self.program_plane.draw(mode = gl.GL_TRIANGLE_STRIP, indices = self.plane_indies)

    def print_help():
        print('''Simple Point Visualizer Based on Vispy\nUsage: python point_visualizer.py data_file\n[Space] - toggle animation\n[F] - toggle fullscreen\n[Esc] - exit\n''')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print_help();
        pass
    else:
        c = Canvas(sys.argv[1])
        c.show()
        print_help();
        app.run()
